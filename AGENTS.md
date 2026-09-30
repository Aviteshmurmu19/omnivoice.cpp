# AGENTS.md — omnivoice.cpp

Fork of [`ServeurpersoCom/omnivoice.cpp`](https://github.com/ServeurpersoCom/omnivoice.cpp)
that adds GitHub Actions CUDA builds.

## 0. TL;DR for Agents

- What it is: a C++17/GGML port of OmniVoice (k2-fsa/Xiaomi, Apache 2.0), a
  non-autoregressive mask-predict TTS over a Qwen3 0.6B backbone with a
  separate audio tokenizer. Three modes: voice cloning, voice design, auto
  voice. 646 languages, 24 kHz mono, 25 fps codebooks, 8 RVQ codebooks.
- This file is a snapshot of **intent**, not a log or a mirror of the code.
  After a task, edit the section a change actually affects — never append a
  dated summary block.

Five things to know before touching anything:

1. **`ggml/` is a mandatory submodule** and it is `ServeurpersoCom/ggml`, not
   `ggml-org/ggml`. It carries two custom ops the codec cannot build without:
   `GGML_OP_SNAKE` and `GGML_OP_COL2IM_1D`. Repointing `.gitmodules` silently
   changes which code is compiled.
2. **`src/omnivoice.h` is the only public ABI.** `tests/abi-c.c` is compiled on
   *every* build with `-std=c99 -Wall -Werror -pedantic`, so a change that
   breaks plain-C consumability fails the main build, not an opt-in step.
3. **CI has no GPU.** A green run proves compile + package + start, never
   synthesis. Both workflows are `workflow_dispatch` only — pushing is always
   safe and never triggers a build.
4. **Almost every `src/*.h` is a deliberate port of PyTorch reference code.**
   Bit-parity with the Python model is the contract. "Cleaner" rewrites that
   change reduction order, rounding, or dtype are regressions, even when the
   new code looks better.
5. **CI facts are duplicated in four places** that must move together — see
   §8 "Keep documentation in sync".

Where the real logic lives (pointers, not code):

| Concern | File |
|---|---|
| Full TTS orchestration (single-shot, chunked, streaming, cancel) | `src/pipeline-tts.cpp` |
| Audio tokenizer encode/decode | `src/pipeline-codec.cpp` |
| Public `ov_*` ABI implementation | `src/omnivoice.cpp`, decl in `src/omnivoice.h` |
| Prompt + CFG batch construction | `src/prompt-tts.h` |
| MaskGIT iterative decoder | `src/maskgit-tts.h` |
| Build system / targets | `CMakeLists.txt` |
| CI build matrix | `.github/workflows/build-cuda.yml` |
| Model + pipeline reference doc | `docs/ARCHITECTURE.md` |
| Per-module contracts (nearest file wins) | `src/AGENTS.md`; harnesses in `tests/AGENTS.md` |
| CUDA build decision record | `docs/superpowers/specs/2026-09-29-omnivoice-cuda-build-design.md` |

## 1. Mental Model & Architecture

**Text → audio, in one pass of the pipeline:**

```
target text (stdin/param)
  -> BPE tokenize (src/bpe.h) + prompt build (src/prompt-tts.h)
       [<|lang_start|>..][<|instruct_start|>..][<|text_start|>ref+target<|text_end|>][ref codes][MASK x T]
  -> MaskGIT loop (src/maskgit-tts.h): num_step (default 32) full bidirectional
       prefill of the Qwen3 LM, cond+uncond CFG batched as B'=2, progressive
       demasking -> int32 codes [K=8, T]
  -> RVQ decode + fc2 + DAC decoder (src/rvq-codec.h, src/dac-decoder.h)
  -> post-processing (src/audio-postproc.h) -> WAV (src/audio-io.h)
```

**Reference (clone) path, audio → codes:**

```
ref wav 24 kHz -> resample 16 kHz (src/audio-resample.h) -> pad 160 each side
  -> HuBERT (src/hubert-enc.h) -> mean of 13 hidden states -> /2
  -> SemanticEncoder (src/semantic-enc.h)      [768, T]
  -> DAC encoder (src/dac-encoder.h)           [256, T]
  -> concat(1024) -> fc -> RVQ encode          [8, T] @ 25 fps
```

Key facts about the flow:

- There is **no KV cache**: attention is fully bidirectional, so prefix hidden states drift as tokens unmask. Every MaskGIT step is a full prefill at cost `2 * forward(B, S)` (the 2 = CFG cond + uncond rows).
- Chunking: text longer than `chunk_threshold_sec` (30 s estimated) splits on punctuation into `chunk_duration_sec` (15 s) chunks, cross-faded 0.3 s. In auto-voice, chunk 0's tokens become the voice prompt for chunks 1..N.
- Determinism: with `class_temperature == 0` **and** `position_temperature == 0` the decoder is bit-deterministic; otherwise Philox4x32-10 (`src/philox.h`, seedable) is used and its counter is threaded across chunks to match the reference's single global RNG.

**Modules, one line each:**

| Module | Responsibility |
|---|---|
| `src/omnivoice.{h,cpp}` | Opaque `ov_context` handle; the C ABI boundary; catches all C++ exceptions |
| `src/pipeline-tts.{h,cpp}` | LM load, forwards, long-form/streaming synthesis, cancel, post-proc |
| `src/pipeline-codec.{h,cpp}` | Audio tokenizer end-to-end encode and decode |
| `src/qwen3-enc.h`, `src/omnivoice-llm.h` | Transformer blocks / LM weight holder |
| `src/prompt-tts.h`, `src/maskgit-tts.h` | Prompt+CFG batching / iterative decoder |
| `src/rvq-codec.h`, `src/dac-*.h`, `src/hubert-enc.h`, `src/semantic-enc.h` | Audio tokenizer stages |
| `src/gguf-weights.h`, `src/weight-ctx.h`, `src/backend.h`, `src/static-graph.h` | GGUF mmap+load, weight staging, backend pair, graph alloc strategy |
| `src/bpe.h`, `lang-map.h`, `voice-design.h`, `text-chunker*.h`, `duration-estimator.h` | Text-side Python ports |
| `src/audio-*.h`, `wav.h`, `srt.h` | Audio IO, resampling, post-processing, SRT parsing |
| `src/tts-server.h` + `tools/tts-server.cpp` | OpenAI-compatible HTTP server |
| `tools/*.cpp` | CLIs (`omnivoice-tts`, `omnivoice-codec`, `tts-server`, `quantize`) |

## 2. Layout Map

```
CMakeLists.txt        build: 1 static core lib, optional shared lib, 4 exes, 1 C ABI test
src/                  all implementation. Almost every .h is header-only + `static` functions;
                      only omnivoice.cpp, pipeline-tts.cpp, pipeline-codec.cpp are compiled
src/AGENTS.md         per-module contracts (exact identifiers, invariants, gotchas)
tools/                CLI entry points + version.cmake (git hash -> version.h); AGENTS.md
tests/                abi-c.c (built by default); *.py cossim harnesses; *.log reference runs;
                      NESTED AGENTS.md (harness contract + what the baseline logs prove)
examples/             tts/clone/server scripts + freeman.{wav,rvq,txt} reference triple;
                      AGENTS.md (per-script cwd + fixtures)
docs/                 ARCHITECTURE.md (pipeline reference), superpowers/specs/ (decision records)
vendor/               cpp-httplib (HTTP, no SSL), yyjson (JSON); AGENTS.md (do not edit)
ggml/                 SUBMODULE -> github.com/ServeurpersoCom/ggml.git
.github/workflows/    build-cuda.yml, publish-artifacts-to-release.yml, + their own AGENTS.md
*.sh / *.cmd          build + model helper scripts (run from repo root)
```

Trivial/no-surprise files (one line each): `.clang-format` (LLVM-lineage, C++17,
`ColumnLimit: 120`, `IndentWidth: 4`, `PointerAlignment: Middle`, LF endings);
`.gitattributes`; `LICENSE` (MIT); `update.{sh,cmd}` (pull submodule then repo).

> `.github/workflows/AGENTS.md` is a **nested AGENTS.md** — it takes precedence
> for anything inside `.github/workflows/`. Read it before editing a workflow.

## 3. Public Surface — Contracts, Not Implementations

### 3.1 C ABI — `src/omnivoice.h`

Single header, pure C99, `extern "C"`, POD structs only. `OV_ABI_VERSION` is
**3** and is the only compat number (there is no semver triple).

| Entry | Signature shape | Contract |
|---|---|---|
| `ov_version` | `const char * (void)` | `"<git-hash> (<date>)"`, static, never NULL |
| `ov_last_error` | `const char * (void)` | **thread-local**, errno-style: meaningful only right after a failure; empty string = no error |
| `ov_init_default_params` | `(struct ov_init_params *)` | `use_fa=true`, `clamp_fp16=false`, `codec_path=NULL`, stamps `abi_version` |
| `ov_init` | `ov_context * (const struct ov_init_params *)` | NULL on failure after unwinding; handle owns its backend pair |
| `ov_free` | `(ov_context *)` | Safe on NULL and on partially initialised state |
| `ov_tts_default_params` | `(struct ov_tts_params *)` | `chunk_duration_sec=15`, `chunk_threshold_sec=30`, `denoise=true`, `preprocess_prompt=true`, `mg_num_step=32`, `mg_guidance_scale=2.0`, `mg_t_shift=0.1`, `mg_layer_penalty_factor=5.0`, `mg_position_temperature=5.0`, `mg_class_temperature=0.0`, `mg_seed=42`, `postproc=true`, all strings/pointers NULL |
| `ov_synthesize` | `(ov_context *, const ov_tts_params *, ov_audio * out)` | Fills `out` with mono f32 @ 24 kHz; **requires a codec-loaded handle** |
| `ov_duration_sec_to_tokens` | `int (const ov_context *, float)` | Frames at `sample_rate / hop_length`, clamped to ≥1; returns **1** (not 0) on error |
| `ov_num_codebooks` | `int (const ov_context *)` | K; 0 for NULL handle |
| `ov_n_languages` / `ov_language_id` / `ov_language_name` | `int` / `const char * (int)` | Static table; id like `"fr"`, name like `"french"`; out-of-range → NULL |
| `ov_extract_voice_ref` | `(ov_context *, const float *, int, ov_voice_ref *)` | Same preprocessing as the CLI encode → bit-identical codes |
| `ov_audio_free` / `ov_voice_ref_free` | `(struct *)` | Free the `malloc`'d buffer and zero the struct; safe on zero-init |
| `ov_log_set` | `(ov_log_cb, void *)` | Process-wide (not per handle); NULL restores stderr |

Ownership rules (load-bearing):

- `ov_audio.samples` and `ov_voice_ref.ref_codes` are **`malloc`-allocated**, so
  C bindings can `free()` them without the C++ runtime. Never `delete[]`.
- Both `ov_synthesize` and `ov_extract_voice_ref` **free the output struct on
  entry** → pass a zero-initialised (`= {0}`) or previously valid struct, never
  garbage.
- `params->abi_version > OV_ABI_VERSION` is rejected; older versions accepted.
  `postproc` is read only when `abi_version >= 3`.
- Exceptions never cross the boundary: any `std::exception` becomes
  `ov_set_error(...)` + a negative `ov_status`.
- `enum ov_status`: `0, -1 INVALID_PARAMS, -2 INSTRUCT_INVALID,
  -3 GENERATE_FAILED, -4 OOM, -5 CANCELLED`.
- Cancellation granularity is ~`chunk_duration_sec`: `cancel` is polled between
  chunks, not inside MaskGIT steps.

### 3.2 `omnivoice-tts` — `tools/omnivoice-tts.cpp`

```
omnivoice-tts --model <gguf> --codec <gguf> [options] -o <out.wav> < text.txt
```

- **Target text always comes from stdin** — there is no text flag.
- **`--ref-text` is a file path**, never inline text. Required whenever
  `--ref-wav` or `--ref-rvq` is given.
- `-o -` streams a WAV to stdout (incremental stdin read, synthesis starts at
  the first sentence boundary). Everything else — usage, errors, seed,
  progress — goes to **stderr**.
- Exit codes: `0` success, `1` everything else (including caught exceptions).

| Flag | Default | Notes |
|---|---|---|
| `--model <gguf>` | required | LLM GGUF, all modes |
| `--codec <gguf>` | required in TTS mode | not needed for `--llm-test` / `--maskgit-test` |
| `-o <path>` | required | `-` = stdout stream |
| `--format` | `wav16` | `wav16` \| `wav24` \| `wav32` |
| `--lang <str>` | `""` | case-insensitive name or lowercase ISO id; resolved via `src/lang-map.h` |
| `--instruct <str>` | `""` | voice-design attributes, validated by `src/voice-design.h` |
| `--duration <sec>` | `0` = estimate | `>0` sets `T_override`, forces single-shot, bypasses chunker |
| `--no-denoise` | denoise on | |
| `--ref-wav <path>` / `--ref-rvq <path>` | none | mutually exclusive |
| `--ref-text <path>` | — | file path |
| `--seed <int>` | `-1` → `std::random_device` | printed as `[CLI] Seed: %llu` |
| `--steps <int>` | `0` → library default 32 | `<1` is an error |
| `--no-preprocess-prompt` | preprocess on | |
| `--chunk-duration <sec>` | `15.0` | `<= 0` disables chunking |
| `--chunk-threshold <sec>` | `30.0` | |
| `--stream-by-line` | off | one RIFF header per line, only with `-o -` |
| `--srt <path>` | none | dubbing; per-cue duration from the SRT; incompatible with `-o -`, `--stream-by-line`, `--duration` |
| `--no-fa`, `--clamp-fp16`, `--dump <dir>` | off | debug |
| `--llm-test <input.bin>`, `--maskgit-test` | off | debug modes, mutually exclusive, bypass `ov_init` |

Debug-mode binary formats: `--llm-test` input `[i32 K, i32 S, K*S ids, S mask]`,
output `[i32 V, i32 K, i32 S, V*K*S f32]`; `--maskgit-test` output raw
`int32 [K, T]` row-major, greedy, codec not loaded.

### 3.3 `omnivoice-codec` — `tools/omnivoice-codec.cpp`

```
omnivoice-codec --model <codec-gguf> -i <input> [--format wav16|wav24|wav32]
```

Mode is inferred from the **input extension** (`.wav` → encode, `.rvq` →
decode); there is no output flag — the output is auto-named by swapping the
extension (`clip.wav` → `clip.rvq`). `RVQ_CODE_BITS = 11` is hard-coded in all
three tools and must match. Encode applies the same preprocessing as the TTS
`--ref-wav` path (RMS auto-gain, silence trim, hop truncation), so the resulting
`.rvq` feeds `omnivoice-tts --ref-rvq` bit-identically.

### 3.4 `tts-server` — `tools/tts-server.cpp` + `src/tts-server.h`

```
tts-server --model <gguf> --codec <gguf> [--host 127.0.0.1] [--port 8080]
           [--lang None] [--no-fa] [--clamp-fp16]
```

| Endpoint | Contract |
|---|---|
| `POST /v1/audio/speech` | OAI TTS. `response_format`: `"pcm"` (default, s16le 24 kHz mono, chunked as generated) or `"wav"` (one-shot RIFF). Fields: `language` (unknown → 400), `voice`, `instructions`, `seed`, `response_format` |
| `GET /v1/models` | the single loaded model; id = basename of `--model` |
| `GET /v1/audio/voices` | registered voices |
| `POST /v1/audio/voices` | `{name, ref_text, wav_b64}` (encoded server-side) **or** `{name, ref_text, rvq_b64}` (verbatim) |
| `DELETE /v1/audio/voices/{name}` | drop a voice |
| `GET /health` | liveness |

One GPU-resident context; synthesis serialised by `g_synth_mutex`, registry by
`g_voices_mutex`. Unknown voice name → `OV_STATUS_INVALID_PARAMS`, never a
silent fallback to voice design. **Registered codes carry no loudness**: a
registered voice always lands on the `ref_rms < 0` branch and gets peak
normalisation instead of reference-level matching.

### 3.5 `quantize` — `tools/quantize.cpp`

`quantize <input.gguf> <output.gguf> <type>` — exactly 4 args. Types
(case-insensitive): `BF16 Q2_K Q3_K_S Q3_K_M Q3_K_L Q4_K_S Q4_K_M Q5_K_S
Q5_K_M Q6_K Q8_0`. Quantization policy lives entirely in `should_quantize`:
RVQ codebooks (`quantizer.quantizers.*`), `.snake*.alpha`, `fc.weight`,
`fc2.weight`, 1-D tensors and embed tables are never K-quantised (embeddings
get an `embed` variant because CUDA `ggml_get_rows` rejects K-quants). Rows
whose `ne[0]` is not a multiple of the block size fall back to F16.

### 3.6 Low-level API — `src/pipeline-tts.h`, `src/pipeline-codec.h`

Direct access to `pipeline_tts_llm_forward[_batched]`, `pipeline_tts_generate`,
pipeline_codec_encode/decode`, `pipeline_tts_resolve_instruct`,
`pipeline_tts_load` / `pipeline_codec_load` / `backend_init`. Used by the CLI
debug modes and the Python cossim harnesses. **Intentionally not part of the
public ABI** (C++ types in signatures, no visibility export) — recommend
`ov_*` to everything else.

## 4. Module Reference

Per-module contracts — exact identifiers, invariants, gotchas for every
`src/*` file — live in **`src/AGENTS.md`**: agents read the nearest AGENTS.md
in the directory tree, so each source module carries its contract next to the
code. One-line roles are in §1; system-wide invariants (load order, dtype
policy, determinism, thread-safety) are in §10. Entries state contract,
invariant and gotcha — never the implementation.

Everything outside `src/` keeps its contract here:

### `tools/*`, `convert.py`, scripts
- `tools/version.cmake`: emits `#define OMNIVOICE_VERSION "<hash> (<date>)"`
  via `git rev-parse --short HEAD` + `git show -s --format=%cs HEAD`, with an
  `"unknown"` fallback → `fetch-depth: 0` is not needed in CI.
- `convert.py`: safetensors → 2 GGUFs, **takes no CLI arguments** (inputs and
  outputs are hardcoded to `checkpoints/OmniVoice/` and `models/`). Skips if
  outputs exist; exits 1 with `run checkpoints.sh first` when the checkpoint
  dir is missing. Dtype policy is passthrough (F32/BF16/F16 only); it folds the
  HuBERT `weight_norm` at convert time and writes `omnivoice.special.*` KVs.
- `checkpoints.sh` (`hf download k2-fsa/OmniVoice`), `models.sh`
  (`hf download Serveurperso/OmniVoice-GGUF`), `quantize.sh`, `update.{sh,cmd}`,
  `format.sh` — all assume **repo root** as cwd. All `.sh` build scripts wipe
  `build/` first; the `.cmd` twins do not (the `rd` line is commented out).
- `buildcuda.sh` hardcodes `/usr/local/cuda/bin/nvcc` and sets **no**
  `CMAKE_CUDA_ARCHITECTURES`; `buildall.*` adds
  `GGML_CPU_ALL_VARIANTS=ON GGML_CUDA=ON GGML_VULKAN=ON GGML_BACKEND_DL=ON`.

### `.github/workflows/*`
See the nested `.github/workflows/AGENTS.md` before editing. Summary:
`build-cuda.yml` (`workflow_dispatch`, 8 jobs, `fail-fast: false`) and
`publish-artifacts-to-release.yml` (`workflow_dispatch`, inputs `run_id`,
`tag`, `pattern='omnivoice-'`). Both are push-inert by design.

## 5. Data Models / Schemas

### GGUF files (produced by `convert.py`, quantised by `tools/quantize.cpp`)

| File | `general.architecture` | Carries |
|---|---|---|
| `omnivoice-base-{F32,BF16,Q8_0,Q4_K_M}.gguf` | `omnivoice-lm` | Qwen3 28-layer backbone + `audio_embeddings` + `audio_heads` + BPE tokenizer + `omnivoice.special.*` |
| `omnivoice-tokenizer-{F32,BF16,Q8_0,Q4_K_M}.gguf` | `omnivoice-tokenizer` | HuBERT + SemanticEncoder + DAC enc/dec + RVQ + `fc`/`fc2` + sample-rate/hop metadata |

Key metadata (see `docs/ARCHITECTURE.md` §GGUF layout for the full tensor list):
`block_count 28`, `embedding_length 1024`, `head_count 16`, `head_count_kv 8`,
`vocab_size 151676`, `omnivoice.num_audio_codebook 8`,
`omnivoice.audio_vocab_size 1025`, `omnivoice.audio_mask_id 1024`,
`omnivoice.sample_rate 24000`, `omnivoice.acoustic.hop_length 960`.

**Quantisation policy:** Q8_0 (or lower) on the base LM only. The tokenizer
GGUF keeps `quantizer.quantizers.*`, `fc`, `fc2` and `.snake*.alpha` at source
dtype in **every** variant — nearest-neighbour codebook lookup is sensitive to
per-row noise and even BF16 truncation drifts codes enough to break cloning.

### `.rvq` binary container
Headerless. See `src/rvq-file.h` in §4. `K` from the codec GGUF,
`code_bits = 11` hard-coded in all three tools.

### `struct ov_audio` / `struct ov_voice_ref`
`ov_audio { float* samples; int n_samples; int sample_rate /*24000*/; int channels /*1*/ }`.
`ov_voice_ref { int32_t* ref_codes; int ref_T; int num_codebooks }` laid out
`[num_codebooks, ref_T]` row-major (T fastest). Both `malloc`-owned, both
released only by their `_free` function.

### Debug dump format (`src/debug.h`)
`[int32 ndims][int32 dim0..dimN-1][float data…]`, native endian, no magic,
path `<dir>/<name>.bin`.

## 6. Configuration Reference

| Key / Env Var | Type | Default | Effect |
|---|---|---|---|
| `GGML_BACKEND` | env | auto-best | Force a device by name (`CUDA0`, `Vulkan0`, `CPU`, `BLAS`); unknown name → empty `BackendPair` + `[Load]` error listing available |
| `OMNIVOICE_DUMP_STAGES` | env | unset | Codec decode/encode dumps `cpp_*.raw` into **CWD** (distinct from `--dump`/`dump_dir`) |
| `OMNIVOICE_LOOP_FORWARD` | env | unset | Forces the per-row LM forward loop; also changes batched-forward output to full `B'*V*K*S` layout |
| `GGML_CUDA_DISABLE_GRAPHS` | env | unset | Runtime off switch for CUDA graphs (see `CMakeLists.txt`) |
| `GGML_CUDA_GRAPHS` | CMake | `ON` | CUDA graphs on by default here (standalone ggml ships them off) |
| `GGML_SOURCE_DIR` | CMake | `<repo>/ggml` | ggml source tree; `add_subdirectory(${GGML_SOURCE_DIR} ggml)` |
| `OMNIVOICE_SHARED` | CMake | `OFF` | Build `libomnivoice.so`/`.dll` exporting only `ov_*`, everything else `-fvisibility=hidden` |
| `CMAKE_CUDA_ARCHITECTURES` | CMake | see `CMakeLists.txt` | If unset, defaults to `75-virtual;80-virtual;86-real;89-real` or, with toolkit ≥12.8, adds `120a-real;121a-real`. CI always pins it explicitly |
| `GGML_MAX_NAME` | compile def | forced `128` | Audio-tokenizer tensor names exceed ggml's default 64 |
| `GGML_CPU_ALL_VARIANTS`, `GGML_BLAS`, `GGML_SYCL`, `GGML_VULKAN` | CMake | off | Other backend scripts |

## 7. Canonical Patterns

**Add a field to `ov_tts_params` (ABI-safe growth):**
```
1. Append the field at the END of the struct (never insert mid-struct).
2. Bump OV_ABI_VERSION in src/omnivoice.h.
3. Initialise it in ov_tts_default_params (src/omnivoice.cpp).
4. Read it only behind `if (params->abi_version >= <new>)`.
5. Extend tests/abi-c.c if the default is part of the contract.
```

**Load a new GGUF tensor into the LM:**
```
In:  gf_load_tensor(&wctx, gf, "exact.tensor.name")        -> deferred copy
     gf_load_tensor_f32(...)                               -> cast BF16/F16 -> F32 at load
Then: wctx_alloc(bp.backend) MUST run before gf_close()
Sizing: bump the wctx_init(n_tensors) argument in pipeline-tts.cpp / pipeline-codec.cpp
```

**Make a graph debuggable:** call `ggml_set_output` on the tensor you want to
read — the memory planner will otherwise reuse the buffer and you will dump
garbage. Debug taps in the batched graph rely on this.

**Add a GPU architecture to CI:**
```
1. Add an OBJECT to the matrix `sm` axis: - { cuda: '120a', name: '120a' }
   (`cuda` -> -DCMAKE_CUDA_ARCHITECTURES=<cuda>-real, `name` -> artifact filename)
2. Probe locally first, in seconds:
   cmake -S . -B /tmp/probe -DCMAKE_CUDA_ARCHITECTURES=120a-real
3. Check tensor cores before touching GGML_CUDA_FORCE_MMQ (Turing-only today).
4. Add the row to README.md *GPU support* and the matrix YAML block.
```
`sm_60` is deliberately absent: it builds on CUDA 12.x (and is why the toolkit
is not on 13.x) but no runner GPU here is Pascal — add `{ cuda: '60', name: '60' }`
only if a Tesla P100 becomes a target.

## 8. Build / Run / Test / Format

**Setup (mandatory):**
```bash
git clone --recurse-submodules <url>
git submodule update --init --recursive      # if already cloned
git submodule status | grep '^-' && echo "STOP: ggml not initialised"
```
Remotes: `origin` is this fork, `upstream` is the original →
`git pull upstream master`. **This fork's `README.md` replaces upstream's**, so
every upstream edit conflicts there: keep this repo's version and check whether
upstream's README gained anything worth porting.

**Build:**
```bash
./buildcuda.sh        # NVIDIA          -> build/
./buildvulkan.sh      # AMD/Intel       -> build/
./buildcpu.sh         # CPU only        -> build/
./buildall.sh         # all backends + GGML_BACKEND_DL -> build/
buildcuda.cmd         # Windows (vcvars64) -> build/Release/
```
Or manual (matches CI more closely than the scripts do):
```bash
cmake -S . -B build-msvc -G "Visual Studio 17 2022" -A x64 \
  -DGGML_CUDA=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF -DGGML_CUDA_CUB_3DOT2=ON \
  -DCMAKE_CUDA_ARCHITECTURES=75-real
cmake --build build-msvc --config Release -j %NUMBER_OF_PROCESSORS%
```

**Output locations** — `CMakeLists.txt` sets `CMAKE_RUNTIME_OUTPUT_DIRECTORY`
with a plain `set()`, so `-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=...` is **ignored**:
- Windows, MSVC multi-config: `build-msvc/Release/`
- Linux, Ninja/single-config: `build/`

**Models** (not in the repo — `models/*.gguf` is gitignored, and models are not
in the CI artifacts):
```bash
./checkpoints.sh      # hf download k2-fsa/OmniVoice -> checkpoints/OmniVoice/
python convert.py      # -> models/omnivoice-base-F32.gguf + omnivoice-tokenizer-F32.gguf
./quantize.sh          # -> BF16 / Q8_0 / Q4_K_M variants
./models.sh            # OR just download prebuilt GGUFs from Serveurperso/OmniVoice-GGUF
```
`models.sh` and `quantize.sh` are **alternative** workflows: `quantize.sh`
needs `convert.py` output, `models.sh` downloads finished files.

**Run:**
```bash
echo "Hello world." | ./build/omnivoice-tts \
  --model models/omnivoice-base-Q8_0.gguf \
  --codec models/omnivoice-tokenizer-Q8_0.gguf --lang English -o hello.wav
./build/tts-server --model ... --codec ... --port 8080   # examples/server.sh
```
See §3.2 for the full flag contract. Verified baseline: GTX 1650, driver
610.74, Q8_0 weights, cross-language clone, 4.96 s audio, **RTF 4.4**, 23 s
wall clock, `exit 0` — measured *before* `GGML_CUDA_FORCE_MMQ=ON`; re-measure
before quoting it. No TPU support (ggml has no TPU/XLA backend).

**Test / verify:**
- `test-abi-c` is built by default and is the ABI gate (exit 0–9, see
  `tests/abi-c.c`). It never loads a model.
- No ctest suite. Parity is proven by `tests/debug-tts-cossim.py` and
  `tests/debug-clone-cossim.py` (per-stage cosine sim vs the PyTorch
  reference) and `tests/cross-decode.py` — read **`tests/AGENTS.md`** for how
  to run them, what the committed baseline logs actually show, and the fact
  that a harness run asserting no threshold can exit 0 while diverging.
- **CI runners have no GPU.** The `--help` smoke test is `continue-on-error`
  and passes with missing DLLs. A green run proves compile + package + start
  only — always confirm a real WAV on hardware.
- Formatting: `./format.sh` runs `clang-format -i` over `src/`, `tools/`,
  `tests/` (excludes `build/`, `ggml/`, `vendor/`). Config in `.clang-format`.

**CI budget:** Windows ~30 min, Linux ~15 min, four architectures in parallel.
Windows artifact ~564 MB (almost entirely `cublasLt64_12.dll`), Linux ~21 MB.

**Publish artifacts to a release** (the only deployment path — there is no
CD). Dispatch `.github/workflows/publish-artifacts-to-release.yml` with the
inputs below, or drive it from a terminal:

```bash
gh workflow run publish-artifacts-to-release.yml \
  -f run_id=<build-cuda run id> -f tag=v0.1.0 -f pattern=omnivoice-
# then delete the run's artifacts: gh api -X DELETE .../actions/runs/<id>/artifacts
```

`run_id` comes from the `build-cuda` run, `tag` must already exist as a
release, `pattern` defaults to `omnivoice-`. The job has no checkout, so it
passes `--repo` to every `gh` call — keep that. Models are never part of a
release: they live on Hugging Face (`models/*.gguf` is gitignored).

**Keep documentation in sync.** Flags, runners, architectures and artifact
names are described in four places — a change to one that isn't reflected in
the others is drift, and it is how the next agent "fixes" something
deliberately pinned:

| Where | What it claims |
|---|---|
| `.github/workflows/build-cuda.yml` inline comments | why each flag exists |
| `README.md` → *Design notes*, *GPU support* | pinned CUDA, `windows-2022`, the arch table, artifact names |
| this file → *CI traps* | the failure modes, for agents |
| `docs/superpowers/specs/2026-09-29-omnivoice-cuda-build-design.md` | the decision record, with sources (spec is source of truth for *why*) |

Update all four in the same commit. If you can't cite where a decision came
from, flag it as unverified in the spec rather than stating it as fact.

**Do not copy upstream's CI.** Upstream has no working CI anywhere (here,
`/.github/workflows` 404s): the `release.yml` shared by `acestep.cpp`,
`minimaxmusic.cpp`, `yue2.cpp` and `s2s.cpp` has **never executed** and would
fail anyway — no CUDA version pinned (`Jimver/cuda-toolkit` defaults to
13.2.0), no architecture pin (compiles Maxwell), no CUDA DLLs bundled. Proven
references, most useful first: `niksedk/omnivoice.cpp`,
`handy-computer/transcribe.cpp`, `ggml-org/llama.cpp`, `LostRuins/koboldcpp`,
`withcatai/node-llama-cpp`.

## 9. Conventions

- **Header-only, `static` functions.** Only three `.cpp` files exist under
  `src/`. New model/graph code belongs in a header next to its siblings;
  add it to `omnivoice-core`'s source list only if it needs a TU.
- **Port fidelity over style.** Where a file mirrors a Python module, keep
  names, ordering and constants traceable to the source and say so in the
  header comment. Parity breaks are bugs even when the code compiles.
- **Errors:** internal code throws (`ov_throw`) or returns `bool`/empty
  containers; nothing throws across `extern "C"`. Log with `ov_log(level,
  "[Tag] ...")` — tags are stable prefixes (`[Load]`, `[TTS]`, `[MaskGIT]`,
  `[PipelineCodec]`, `[Quantize]`) used for grepping runs.
- **Ownership:** anything `malloc`'d crosses the ABI and is freed by an
  `ov_*_free`; C++ containers never cross it.
- **Dtype rules:** norms and biases → F32 at load; conv weights → F16 at load;
  everything else keeps the GGUF dtype. Comments saying `bf16` on conv
  tensors are stale.
- **CJK literals** live in `voice-design.h`, `lang-map.h`, `text-chunker.h`;
  MSVC gets `/utf-8` for C/C++ only (nvcc treats a bare `/utf-8` as an input
  file and aborts).
- **Format:** `.clang-format` — 120 cols, 4-space indent, no tabs, LF endings,
  middle-aligned pointers, `BinPackParameters: false`. Run `./format.sh`.
- **Docs:** anything narrative or decision-level goes under `docs/`
  (`docs/superpowers/specs/` for decision records), in this repo.

**Commit messages and PRs:**
- Subject is `<scope>: <imperative summary>`. Observed scopes: `ci:`, `build:`,
  `cmake:`, `docs:`, `abi:`, `tts:`, `codec:`, `server:`, `backend:`, `tests:`,
  `ggml:` (submodule bumps), `script:`, `nits:`.
- The body explains **why**, with evidence: the failing log line, the source
  that established a fact, the measurement that justifies a flag. Commits here
  routinely quote the exact error they fixed (`CMake Error: Unknown argument`).
- Say what was verified ("verified across all four architectures", "RTF 4.4 on
  a GTX 1650") and, when a fix was wrong before, say so. Do not claim a green
  run proves synthesis — CI has no GPU.
- PRs squash to one commit whose title keeps the `<scope>:` form (observed:
  `build: CUDA workflow for Windows and Linux (#1)`). One logical change per
  commit; CI flag changes, README and the design spec move **together**.

## 10. Invariants & Gotchas (system-wide)

**Loading and memory**
- Order is mandatory: `gf_load` → `wctx_init` → loads → `wctx_alloc` →
  `gf_close`. Breaking it reads freed mmap memory. The DAC/HuBERT/semantic
  loaders additionally read the mmap *during* their own load phase.
- Capacity sizes are hand-tuned and must be raised when tensors are added:
  `wctx_init` is 512 (LM) / 64 (codec); the per-module `n_tensors_max` values
  are 256 (DAC enc and dec), 64 (semantic), 32 / 8 / 32 / 4 / 4 (HuBERT feat /
  proj / layer / pos_conv / enc_init). Too small → ggml context OOM inside
  `ggml_new_tensor`.
- `GGML_MAX_NAME` is forced to `128` because audio-tokenizer tensor names
  exceed ggml's default 64 — and ggml **silently truncates** at 63 chars.
- Graph node budgets: 8192 (LM), 4096 (codec), 16384 (HuBERT test). Scheduler
  `max_nodes` must be ≥ the graph budget.

**Numerics and parity**
- RoPE is mode 2 (NEOX). Attention is bidirectional and mask-free — it is an
  additive +1.0/0.0 F16 bias, not `-inf`.
- Log-softmax is FP32 by design; FP64 diverges from the reference.
- Two int16 conversions exist on purpose (`*32767` clamped on write,
  `*32768` clipped on the pydub-parity path). Never unify them.
- Determinism needs **both** MaskGIT temperatures at 0; the shipped default
  (`position_temperature = 5.0`) is stochastic but Philox-seeded.

**Concurrency**
- One `ov_context` (and one `PipelineTTS` / `PipelineCodec`) is **not**
  thread-safe: the scheduler is reset and allocated per call and there is no
  mutex. The server serialises with `g_synth_mutex`. Only `g_last_error`
  (thread-local) and the log callback (atomic) are concurrency-safe — and the
  log-dedup state in `backend.h` is not *inferred*.

**Runtime**
- `--ref-text` is a file path; target text is stdin; `-o -` streams to stdout;
  all diagnostics go to stderr.
- `sample_rate`/`hop_length` come from the GGUF, but the ref-encode path
  hard-codes 24000 and the SRT path hard-codes `sr = 24000`.
- `quantize.sh` skips existing outputs, and `quantize.cpp` writes metadata
  then appends data — a failed run leaves a partial file that is then
  permanently skipped *inferred*.

**Security (the server is explicitly not hardened)**
- `tts-server` defaults to `--host 127.0.0.1`, has **no authentication, no
  TLS** (cpp-httplib built without SSL) and no rate limiting. Exposing it past
  localhost means anyone can synthesize and register voices; put an
  authenticating reverse proxy in front or don't expose it.
- Request bodies are capped at `set_payload_max_length(32 * 1024 * 1024)`
  (32 MB) with `set_read_timeout(60)` — that is the only input bound; the base64
  WAV/RVQ uploads are decoded and handed to the same tolerant parsers as the
  CLIs (`wav.h` handles >2 channels by ignoring them, see `src/AGENTS.md`).
- Uploaded `wav_b64` is decoded **server-side and synthesised on the one GPU
  context under the synthesis mutex**, so a large upload stalls every other
  request; registered voice names are attacker-visible via `GET /v1/audio/voices`.
- GGUF loading validates every tensor range against `file_size` before mmap
  use (`gf_load`), which is the truncated-download guard — treat it as load-bearing
  when touching `gguf-weights.h`. Models are still trusted input: a malicious
  GGUF can abort the process through `GGML_ASSERT` paths.

### CI traps

Each of these cost a full 30-minute run to diagnose. Read this subsection
before touching `.github/workflows/`.

**Never use `windows-latest` for a CUDA build.** It ships Visual Studio 2026
and CMake picks the newest generator, so an implicit generator resolves to
`Visual Studio 18 2026` and configure fails with `No CUDA toolset found` at
`enable_language(CUDA)`. Use `windows-2022` **and** pin
`-G "Visual Studio 17 2022" -A x64`. Every project publishing Windows CUDA
binaries pins `windows-2022`; koboldcpp renames the VS 2022 directory to
prevent CMake from selecting it.

**Always pin `CMAKE_CUDA_ARCHITECTURES`.** ggml's default list starts
`50-virtual 61-virtual 70-virtual`, which a CUDA 13.x toolkit cannot compile
at all.

**Stay on CUDA 12.x (pinned `12.8.1`).** CUDA 13 removed offline compilation
for Maxwell/Pascal/Volta and PTX cannot JIT backwards, so a 13.x build cannot
run on a Tesla P100 (`sm_60`) — which Kaggle and Colab gate on. 12.8 is also
the first toolkit that emits `sm_120`, so Blackwell needs no toolkit change.

**Bundle the CUDA DLLs on Windows.** cuBLAS has no static library on Windows
since toolkit 12.3.1, and a driver supplies `nvcuda.dll` but neither cuBLAS
nor cuDart. Ship `cudart64_*`, `cublas64_*`, `cublasLt64_*` next to the
executables or they fail at load with `0xC0000135 STATUS_DLL_NOT_FOUND`
(surfaced as `exit code -1073741515`). Glob versions rather than hardcoding
`_12`, and probe `bin\x64` too — CUDA 13 moved them there. **Linux does not
bundle them**: Colab/Kaggle already ship the runtime.

**The MSVC CRT glob needs two path segments** after `Visual Studio`:
```
$env:ProgramFiles\Microsoft Visual Studio\*\*\VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT\*.dll
```
One `*` silently matches nothing. The artifact shipped green and broken with
only a warning buried in the CUDA install log. **A missing DLL must throw.**

**Derive artifact names from `matrix.os` and `matrix.sm.name`, never a
literal.** A hardcoded `win` against `matrix.os` = `windows` discarded a
29-minute build at the upload step. Keep `if-no-files-found: error` — it is
what caught that.

**`GGML_CUDA_FORCE_MMQ` is Turing-only (`sm_75`), set conditionally.** The
GTX 1650 and T4 have no tensor cores; on Ampere/Ada/Blackwell the flag would
replace cuBLAS tensor-core GEMMs with slower integer dot products. Check tensor
cores before applying it to a new arch.

**The `GGML_*` flags are duplicated verbatim** in the Windows and Linux
`Configure` steps and can silently drift. Five are literal in both —
`GGML_CUDA`, `GGML_BACKEND_DL`, `GGML_NATIVE`, `GGML_CUDA_CUB_3DOT2`,
`CMAKE_CUDA_ARCHITECTURES` — with `CMAKE_BUILD_TYPE=Release` correctly
Linux-only and `GGML_CUDA_FORCE_MMQ` conditional. Verify rather than assume.

**Shell quoting differs between PowerShell and bash, in the direction that
hurts.** An empty PowerShell string still passes an empty argument (CMake
rejects it with `Unknown argument ""`), so the Windows step builds its
conditional flag as an **array** and splats it with `@mmq`. An unquoted empty
bash variable expands to nothing. Same intent, opposite mechanics.

**Backticks do not survive a PowerShell here-string.** `gh release edit` with
an inline `--notes` leaves `\foo\` where `` `foo` `` was meant — use
`--notes-file`. `git commit -m` has the same problem; use `-F <file>`.

**`gh run download` inside a workflow needs `--repo`** — the workflow has no
checkout, so `gh` has no git context to infer the repository from.

**Keep these pre-flight checks before dispatching 8 jobs:**
```bash
git submodule status | grep '^-' && echo "STOP"
for a in 75-real 80-real 89-real 120a-real; do
  cmake -S . -B /tmp/probe-$a -DCMAKE_CUDA_ARCHITECTURES=$a >/dev/null \
    && echo "  $a ok" || echo "  $a FAILED"
  rm -rf /tmp/probe-$a
done
```
Also: the matrix expands to `{os} × {sm}` = 8 jobs; the two Configure steps
still carry identical `GGML_*` literals; each package-step filename matches
the upload glob; storage can absorb the Windows artifacts (or you have a plan
to publish and delete).

**When a job fails:** read the failed step's own log, not the job summary.
`Configure` = toolchain, `Build` = code/architecture, `Package`/`Upload` =
the compile was fine and the *contract* broke. One architecture failing while
others pass points at that architecture's value, not the toolkit or runner.

**Storage is finite:** four Windows artifacts at ~564 MB is 2.3 GB. Publish
to a release (`.github/workflows/publish-artifacts-to-release.yml`) and delete
run artifacts rather than accumulating them.

**`git push` is inert, and deliberately so** — both workflows are
`workflow_dispatch` only. Do not "helpfully" add a `push:` trigger.

## 11. Extension Playbooks

**Before changing a build, a pin, or an artifact contract, ask:**
- Which GPUs must this *actually* run on? Not "all NVIDIA" — the honest list drove every pin here, and an untargeted GPU still breaks the build from the default architecture list.
- What driver will the target machine have? This, not the GPU, constrains the CUDA version; all three machines here clear 580, an older one would forbid CUDA 13.
- Does the target image already provide the runtime? Colab/Kaggle do, a Windows desktop does not — that asymmetry is why Windows bundles cuBLAS.
- Is "no fat binary" a size, debugging, or support concern? Per-arch artifacts cost N× CI minutes but make each failure attributable to one arch.
- Can this be verified without a GPU? If not, say so in the PR rather than letting a green run imply more than it proves.
- Does the change alter the public artifact contract? Renaming a file breaks anyone who scripted a download.

**Add a new public API entry:** declare it in `src/omnivoice.h` with POD
types + `OV_API`, implement it in `src/omnivoice.cpp` inside the
`extern "C"` block, return `enum ov_status`, set errors with `ov_set_error`,
and extend `tests/abi-c.c` so the gate covers it.

**Add a language:** `src/lang-map.h` is auto-generated from upstream
`omnivoice/utils/lang_map.py` — regenerate rather than hand-editing the table.
`ov_n_languages` / `ov_language_id` / `ov_language_name` read it directly.

**Add a voice-design category:** extend `voice_design_init`'s mutually
exclusive sets in `src/voice-design.h` and mirror upstream
`omnivoice/utils/voice_design.py`; the EN/ZH dictionaries must stay paired.

**Add a GPU to CI:** see the recipe in §7, plus the README *GPU support* row.

## 12. What Is NOT Here

- **No TPU/XLA backend** — ggml has none.
- **No `push:` CI trigger, no upstream CI to copy, no ctest suite.**
- **No Python/Rust/Go bindings shipped** — only the C header they would bind
  to; `-DOMNIVOICE_SHARED=ON` builds the exportable shared library.
- **No models in the repo or in artifacts** (`models/*.gguf` gitignored).
- **No GPU in CI**, so no automated synthesis test.
- **No SSL in the HTTP server** (cpp-httplib without TLS; reverse proxy in
  prod) and no auth on any endpoint.
- **No per-request concurrency on one context** — requests serialise.
- **Nested AGENTS.md:** `.github/workflows/AGENTS.md` owns everything in that
  directory (workflow history, invariants, sync table); `src/AGENTS.md`
  owns every per-module contract; `tools/AGENTS.md` the entry-point
  implementation contract; `tests/AGENTS.md` the parity harnesses and what
  the committed baseline logs prove; `examples/AGENTS.md` script cwd and
  fixtures; `vendor/AGENTS.md` the do-not-edit rule. Agents resolve the
  **closest** file in the directory tree. This file owns the rest.

## 13. Known Spec/Code Divergences

- `README.md`/`examples/*.sh` use `omnivoice-tokenizer-Q8_0.gguf` as the
  codec while `models.sh` downloads `omnivoice-tokenizer-F32.gguf` and
  `examples/server.sh` passes the F32 one — both work, but the "recommended"
  codec differs per file.
- README's manual Windows build command sets `GGML_CUDA_FORCE_MMQ=ON`
  unconditionally and omits `GGML_CUDA_CUB_3DOT2` and
  `CMAKE_CUDA_ARCHITECTURES`; the workflow sets MMQ conditionally and pins
  the arch. The workflow is authoritative.
- `docs/superpowers/specs/2026-09-29-...design.md` still shows `windows-latest`
  and artifact `omnivoice-win-cuda-sm75.zip`, and predates the 4-arch matrix;
  the shipped workflow uses `windows-2022` and `omnivoice-windows-*.zip`.
  The spec's status header ("Draft for review", target repo "not yet created")
  is stale.
- `omnivoice-tts --help` claims `--lang`/`--instruct` default to `'None'`;
  the code defaults them to `""`. (`tts-server`'s `--lang None` default *is*
  real.)
- `omnivoice-tts --help` says `-o` produces "24 kHz mono"; the buffered writer
  uses `audio.sample_rate` (24000 in practice, not asserted).
- `--dump <dir>` is parsed in `--llm-test` mode but never passed through to
  that code path.
- Several struct comments describe conv weights as `bf16`; they are allocated
  `GGML_TYPE_F16` (the ARM im2col rule).
- `pipeline-tts.h`/`pipeline-codec.h` say loads leave the struct "in a clean
  state on failure"; early returns free resources but do not zero the struct
  (only the scheduler-failure path calls `_free`).
- `pipeline-codec.h` documents `sample_rate 24000` / `hop_length 960` as
  fixed and decode output as `n_frames * 960`; both are read from GGUF and the
  output length is unchecked.
- `maskgit-tts.h` calls greedy decoding "fully deterministic", but the shipped
  default has `position_temperature = 5.0`.
- `srt_parse` is documented as returning `false` on malformed input; it
  always returns `true`.
- `philox_uniform_fill`'s doc says `[0, 1)`; the formula yields `(0, 1)`.
- `quantize.cpp`'s header says "Reads BF16 GGUF"; it accepts F32/F16/BF16.
- `hubert-enc.h`'s header still says "Stage 1 ports the feature_extractor
  only" although the file contains the full encoder.
- `docs/ARCHITECTURE.md` §Validation quotes `Tokens 100.00% exact`,
  `Audio 0.9999` for **both** the chunked TTS and Clone runs; the committed
  `tests/clone-*.log` baselines top out at `Tokens exact: 19.96%` /
  `Audio: 0.28` (TTS F32 is the only config that reproduces the claimed
  numbers). Do not quote the clone figure without re-running it.
- `docs/ARCHITECTURE.md` §Module map lists `tests/prompt.txt`,
  `tests/ref-audio.wav`, `tests/ref-text.txt` — none exist; the fixtures are
  `examples/prompt.txt` and `examples/freeman.{wav,rvq,txt}`.

## 14. Change Log

- 2026-09-30 — Added `tools/AGENTS.md` (entry-point contract: link matrix,
  exit codes, `RVQ_CODE_BITS` sync), `examples/AGENTS.md` (per-script cwd,
  fixtures) and `vendor/AGENTS.md` (do-not-edit + no-SSL note); updated §2
  and §12.
- 2026-09-30 — Split per-module contracts out to nested `src/AGENTS.md` and
  `tests/AGENTS.md` (§2, §4, §12); root keeps architecture, public surface,
  CI and system-wide invariants only.
- 2026-09-30 — Added commit/PR conventions (§9), deployment steps (§8),
  server security notes (§10), launches nested `tests/AGENTS.md`; corrected
  `wctx_init`/`n_tensors_max` capacity values and abbreviation-set counts.
- 2026-09-30 — Recorded two ARCHITECTURE.md divergences (§13) found by
  diffing its Validation/Module-map claims against `tests/*.log` and the tree;
  added commit-message and PR conventions extracted from `git log`.
- 2026-09-30 — Rebuilt `AGENTS.md` into the full spec structure (§0–14),
  preserving all prior CI/build content and adding the source-level contracts
  (`AGENTS.md`, `src/*`, `tools/*`, workflows, scripts, `convert.py`).
- 2026-09-29 — Initial CI-focused `AGENTS.md` (CUDA build traps, pre-flight,
  GPU architectures) + nested `.github/workflows/AGENTS.md`.

