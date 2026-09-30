# AGENTS.md — `src/`

Per-module contracts for omnivoice.cpp. Read [`../AGENTS.md`](../AGENTS.md)
first: it owns the project-wide architecture (§1), public surface (§3),
system-wide invariants (§10) and CI/build facts (§8). This file owns what each
source module promises **its callers** — contract, exact identifiers,
invariants, gotchas — so a future agent can diff the code against intent
without re-deriving it.

Rules for entries here: state the contract, never paste the implementation;
every identifier must come from the file it describes; mark anything inferred
from behaviour as *inferred*. Update an entry when (and only when) the
module's contract changes — an internal refactor needs no edit.

Two entries below are not contracts but conventions that live with the code:
the dtype policy (§ `gguf-weights.h`) and the load order (§ `weight-ctx.h`);
both are repeated in `../AGENTS.md` §10 because breaking them corrupts memory.

## Entries

### `src/omnivoice.cpp` — ABI implementation
- **Contract:** implements every `ov_*` entry; owns one `BackendPair`,
  `PipelineTTS`, `PipelineCodec` (optional), `BPETokenizer`, `VoiceDesign` per
  handle.
- **Invariant:** load order is `voice_design_init` → `backend_init("LM")` →
  `pipeline_tts_load` → BPE + OmniVoice specials (both from the **same** LM
  GGUF) → optional codec; failure at any step unwinds through `ov_free`, which
  is idempotent on partial state.
- **Gotcha:** `g_last_error` is `thread_local`; `ov_log_set` publishes
  `user_data` before an release-store of the callback pointer. Concurrent
  `ov_synthesize` on one handle is unsafe (shared scheduler, no mutex) —
  *inferred from absence of locking*.

### `src/pipeline-tts.{h,cpp}` — TTS orchestration
- **Contract:** load/free the LM, run single and batched forwards, generate
  tokens, and drive single-shot / chunked / streaming synthesis into `ov_audio`.
- **Exact identifiers:** `wctx_init(&pt->wctx, 512)`,
  `backend_sched_new(bp, 8192)`, `n_max_nodes = 8192`; env
  `OMNIVOICE_LOOP_FORWARD` (forces the per-row forward path and changes the
  returned logits layout); post-proc literals `cross_fade_chunks(..., 0.3)`,
  `remove_silence(audio, sr, 500, 100, 100, -50.0)`, `peak_normalize_half`,
  `fade_and_pad(audio, sr, 0.1, 0.1)`.
- **Invariant:** the attention mask is an **additive F16 bias of +1.0/0.0**, not
  a hard mask — the model was trained against transformers' boolean-mask
  semantics. Never "simplify" it to `-inf`.
- **Invariant:** `shared_ctr_lo` starts at 0 and is shared across every MaskGIT
  call of one request, mirroring PyTorch's continuous RNG.
- **Gotcha:** ref encoding hard-codes `24000` while hop/rate come from GGUF — a
  codec with a different `sample_rate` would silently mismatch.
- **Gotcha:** pre-encoded reference tokens set `ref_rms = -1.0`, i.e. they
  force peak/0.5 normalisation; only raw `--ref-wav` can carry loudness.
- **Gotcha:** `params->postproc` is honoured only at `abi_version >= 3`, and
  the streaming path always post-filters regardless.
- **Gotcha:** cancellation is polled per chunk, not per MaskGIT step — a long
  single-shot run cannot be interrupted mid-generation.

### `src/pipeline-codec.{h,cpp}` — audio tokenizer
- **Contract:** `pipeline_codec_decode(codes [K,T]) -> f32 @ 24 kHz`;
  `pipeline_codec_encode(audio_24k) -> int32 [K,T]`.
- **Exact identifiers:** GGUF keys `omnivoice.sample_rate`,
  `omnivoice.acoustic.hop_length`; tensors `fc.weight`, `fc.bias`,
  `fc2.weight`, `fc2.bias`; `wctx_init(..., 64)`,
  `backend_sched_new(bp, 4096)`, `n_max_nodes = 4096`; env
  `OMNIVOICE_DUMP_STAGES` (dumps `cpp_*.raw` into **CWD**, unlike `dump_dir`
  which writes into the directory — two different mechanisms in one function).
- **Invariant:** the GGUF mmap must stay open across all module loads — DAC,
  HuBERT etc. read raw `gf_get_data` pointers during their own load phase;
  `gf_close` runs only after every module is loaded.
- **Invariant:** HuBERT input is padded `160` zeros each side (`n_16k + 320`);
  the feature-extractor graph does **no** padding itself. Drop the pad and
  frame counts are off by one.
- **Gotcha:** `sample_rate`/`hop_length` are read from GGUF, not fixed at
  24000/960 despite header comments; output length is taken from
  `audio->ne[0]` and never checked against `n_frames * hop_length`.

### `src/prompt-tts.h` — prompt + CFG batch
- **Contract:** builds `input_ids [B', K, S] int32`, `audio_mask [B', S]`,
  `attention_mask [B', S, S]` with `B'=2` (cond then uncond), mirroring
  `_prepare_inference_inputs`.
- **Exact identifiers:** special tokens `<|denoise|>`, `<|lang_start|>`,
  `<|lang_end|>`, `<|instruct_start|>`, `<|instruct_end|>`, `<|text_start|>`,
  `<|text_end|>`, fallback `"None"`; 13 non-verbal tags in
  `PROMPT_NONVERBAL_TAGS` (`"[laughter]"`, `"[sigh]"`, …).
- **Invariant:** `<|denoise|>` is emitted **iff** `denoise && ref_audio_tokens
  != NULL`. Style+text tokens are duplicated across all K codebooks.
- **Gotcha:** `attention_mask` is quadratic in sequence length — long
  `ref_text + text` inflates memory fast. On an early `false` return the
  struct is left from its previous use *inferred* — zero it at the call site.

### `src/maskgit-tts.h` — iterative decoder
- **Contract:** `maskgit_generate(...)` returns flat `int32 [K, T]`
  (k-slow/t-fast) after `num_step` forwards.
- **Exact identifiers:** `MaskgitConfig{num_step=32, guidance_scale=2.0,
  t_shift=0.1, layer_penalty_factor=5.0, position_temperature=5.0,
  class_temperature=0.0, seed=42}`; CFG `lp = c + s*(c - u)`; hardcoded
  top-k class ratio `0.1f`; `lp[mask_id] = -INFINITY` applied *after* the CFG
  log-softmax.
- **Invariant:** log-softmax accumulates in **FP32 on purpose** — FP64 is more
  accurate but diverges from the reference.
- **Gotcha:** the *default* config is **not** deterministic
  (`position_temperature = 5.0 > 0`). Greedy/deterministic means overriding
  *both* temperatures to 0. The header's "greedy is deterministic" sentence is
  conditional.
- **Gotcha:** `maskgit_generate` mutates `prompt->input_ids` in place; calling
  it twice on the same prompt without rebuilding gives wrong results *inferred*.

### `src/omnivoice-llm.h`, `src/qwen3-enc.h` — LM weights and blocks
- **Contract:** hold/validate Qwen3 0.6B weights and build attention/MLP/layer
  graphs. `cfg.is_causal` is forced `false` (bidirectional MaskGIT).
- **Exact identifiers:** `QWEN3_MAX_LAYERS 32`, `OMNIVOICE_MAX_CODEBOOKS 16`;
  GGUF keys `omnivoice-lm.*` (`embedding_length`, `feed_forward_length`,
  `attention.head_count`, `attention.head_count_kv`, `attention.key_length`,
  `block_count`, `rope.freq_base`, `attention.layer_norm_rms_epsilon`) and
  `omnivoice.num_audio_codebook` / `audio_vocab_size` / `audio_mask_id`.
- **Invariant:** **RoPE mode = 2** (NEOX half-split), explicitly not mode 0 —
  marked in-source as "ggml pitfall". Changing it silently destroys parity.
- **Invariant:** `block_count` (1..32) and `num_audio_codebook` (1..16) are
  validated before any tensor load; norms (`input_layernorm`,
  `post_attention_layernorm`, `q_norm`, `k_norm`, `final_norm`) are cast to F32
  at load, projections keep native dtype.
- **Invariant:** projection fusion order is exactly `q,k,v` then `gate,up`
  along the row axis; `ggml_swiglu` splits the first axis of the mul_mat output.
- **Gotcha:** `head_dim` comes from `attention.key_length`, **not** from
  `hidden_size / n_heads`, despite the struct comment saying `D = H / Nh`.
- **Gotcha:** a missing GGUF KV returns **0** from `gf_get_*` with no warning;
  only `block_count` and `num_audio_codebook` are validated.

### `src/rvq-codec.h` — residual VQ
- **Contract:** decode = sum of `project_out(embed[codes])`; encode = residual
  loop with Euclidean nearest-entry search.
- **Exact identifiers:** `RVQ_NUM_CODEBOOKS 8`; KV `omnivoice.codebook_size`,
  `omnivoice.codebook_dim`; tensors `quantizer.quantizers.%d.{codebook.embed,
  project_in.weight, project_in.bias, project_out.weight, project_out.bias}`;
  `embed_sq` is **synthesised at load on CPU**, it is not in the GGUF.
- **Invariant:** biases are loaded F32 because CUDA `ggml_add` only accepts
  src1 in F32/F16. Encode uses `score = 2*embed@h - e_sq` and drops `||h||2`
  as constant across entries.
- **Gotcha:** graph functions are strictly 2-D `[T, 8]` even though comments
  describe `[B, 8, T]`.

### `src/dac-decoder.h`, `src/dac-encoder.h` — DAC codec nets
- **Contract:** decoder `latent [T,256] → audio [960T,1]`; encoder is the
  mirror (`T/960`). Strides `8 5 4 2 3` (=960), 5 blocks, 3 residual units
  with dilations `1 3 9`.
- **Invariant:** **all conv weights are allocated `GGML_TYPE_F16`** — "F16
  mandatory for ARM im2col strict assertion". Several struct comments still
  say `bf16`; they are stale.
- **Invariant:** ConvTranspose1d is decomposed as `mul_mat` + `ggml_col2im_1d`
  over a load-time permuted weight (`(IC, OC, K)` → `dst[(oc*K+k)*IC+ic]`),
  then right-padded by `output_pad = stride % 2`.
- **Gotcha:** `dac_load_alpha`/`dac_load_bias_f32` treat any non-F32 source as
  BF16 — an F16 GGUF silently produces garbage.
- **Gotcha:** the encoder's snake applies **after** the third residual unit,
  not inside it (flagged in-source with `# NOTE`); encoder kernel/pad pairs are
  fixed macros, not `2*stride`.

### `src/hubert-enc.h`, `src/semantic-enc.h` — semantic front end
- **Contract:** HuBERT base (7-layer conv extractor + GroupNorm on layer 0,
  projection, grouped pos-conv, 12 Post-LN layers) → mean of **13** hidden
  states → /2 → `SemanticEncoder` (2 blocks × 2 residual units, ELU, no
  temporal downsample).
- **Exact constants:** `HUBERT_HIDDEN 768`, `HUBERT_NUM_LAYERS 12`,
  `HUBERT_NUM_HEADS 12`, `HUBERT_FFN_INNER 3072`, kernels
  `{10,3,3,3,3,2,2}`, strides `{5,2,2,2,2,2,2}` (cumulative 320), pos-conv
  `k=128, groups=16, pad=64`; `SEM_HIDDEN 768`.
- **Invariant:** layout flips exactly once — extractor is T-first `[T, C]`,
  everything after `feature_projection` is C-first `[C, T]`.
- **Invariant:** pos-conv is decomposed into **16 sub-`ggml_conv_1d` calls**
  because ggml has no grouped conv; output is `T+1` and gets trimmed by view.
- **Gotcha:** HuBERT attention has no mask (bidirectional, Post-LN); a wrong
  layout produces garbage rather than an assert.

### `src/gguf-weights.h`, `src/weight-ctx.h` — GGUF loading
- **Contract:** mmap + validate a GGUF, then either defer byte copies into a
  `WeightCtx` (matrix weights) or transform straight into an already-allocated
  backend tensor (conv/bias).
- **Exact sequence:** `gf_load` → `wctx_init` → N × `gf_load_tensor` →
  `wctx_alloc(backend)` → `gf_close`. Buffer usage is
  `GGML_BACKEND_BUFFER_USAGE_WEIGHTS` so the scheduler assigns ops by weight
  location.
- **Invariant:** `PendingCopy::src` points **into the mmap** — `wctx_alloc`
  must run before `gf_close`, or you read freed memory.
- **Gotcha:** missing tensor → `ov_throw` (exception), *except*
  `gf_try_load_tensor` and the fused pair loaders which return `NULL` to
  trigger fallbacks. Missing KV → **0, silently**.
- **Gotcha:** `gf_load` verifies every tensor range fits `file_size` — this is
  the truncated-download guard; don't remove it.
- **Gotcha:** `gf_load_qkv_fused` shape mismatch is a `GGML_ASSERT` (abort),
  type mismatch is a `NULL` return (fallback).

### `src/backend.h`, `src/static-graph.h` — backends and graphs
- **Contract:** `backend_init(label)` returns the best device + a CPU
  fallback; `static_graph_*` picks direct `gallocr` when every node is
  supported, else the scheduler, and remembers which so compute uses the
  matching API.
- **Exact identifiers:** env **`GGML_BACKEND`** forces a device by name
  (`CUDA0`, `Vulkan0`, `CPU`, `BLAS`); failure returns `BackendPair{}` with
  `.backend == NULL` — **callers must check before any `pipeline_*_load`**.
  `max_nodes` guidance in-source: 4096 small / 8192 large.
- **Invariant:** on a CPU-only machine `bp.backend == bp.cpu_backend` (same
  pointer) and `backend_release` frees once.
- **Gotcha:** ggml backends are loaded exactly once (function-local `static`
  guard) but the log-dedup state in `ov_ggml_log` is **not** thread-safe
  *inferred*. A forced `GGML_BACKEND=CPU` re-inits through `cpu_backend_new`,
  discarding the forced instance *inferred*.

### `src/bpe.h` — tokenizer
- **Contract:** Qwen2/GPT-2 byte-level BPE loaded from GGUF keys
  `tokenizer.ggml.tokens` / `tokenizer.ggml.merges`, plus a manual port of the
  GPT-2 pre-tokenizer regex.
- **Invariant:** specials (`omnivoice.special.denoise|lang_start|lang_end|
  instruct_start|instruct_end|text_start|text_end`) are matched verbatim and
  bypass BPE; leftmost match wins. `bpe_encode(..., add_eos)` appends
  `eos_id` only when `eos_id >= 0`.
- **Gotcha:** the pre-tokenizer is a hand-rolled approximation of `\p{L}` /
  `\p{N}`; exotic scripts and U+200B-as-whitespace can diverge from Python
  *inferred*. Unmatched raw bytes are dropped with only a warning.

### `src/lang-map.h`, `src/voice-design.h` — language and instruct
- **`lang-map.h`:** auto-generated from `omnivoice/utils/lang_map.py`. Lookup
  contract: **names are ASCII-case-insensitive** (`"Hindi"` → `"hi"`);
  **ISO ids are matched case-sensitively against the raw input**, so ids must
  already be lowercase (`"EN"` → `""`). `"none"`/empty/unrecognised → `""`.
- **`voice-design.h`:** validates the attribute string against 6 mutually
  exclusive categories (gender, age, pitch, style, accents, dialects) and
  resolves EN↔ZH. Suggestion threshold `best_ratio = 0.6f`; separator is `", "`
  for English and `","` when any item contains CJK. Accent forces English,
  dialect forces Chinese, and having both is an error.
- **Gotcha:** both files lazy-init static maps on first call with **no lock** —
  concurrent first use races *inferred*.

### `src/text-chunker.h`, `src/text-chunker-stream.h`, `src/duration-estimator.h`
- **Contract:** sentence split/merge on punctuation, its incremental twin, and
  the `RuleDurationEstimator` port that turns text into an estimated frame
  count. All lengths are **UTF-8 codepoints**, matching Python `len(str)`.
- **Exact identifiers:** `OMNIVOICE_MIN_CHUNK_LEN = 3` (the single source of
  truth for buffered *and* streaming paths); 49-entry abbreviation set in
  `chunker_abbreviations()`. The 10-accent / 12-dialect sets live in
  `voice-design.h`, not here.
- **Invariant:** `text_chunker_stream::flush_eof()` emits chunks **identical**
  to the offline `chunk_text_punctuation(text, chunk_len, OMNIVOICE_MIN_CHUNK_LEN)`
  — the offline call inside the stream wrapper deliberately passes
  `min_chunk_len = 0` and re-implements folding; don't "fix" that call site.
- **Gotcha:** streaming holds back one chunk of look-ahead; equivalence is
  claimed only for `min_chunk_len == 3`.
- **Gotcha:** `duration_estimate_tokens` falls back to the anchor
  `"Nice to meet you."` at 25 tokens when there is no reference.

### `src/audio-io.h`, `src/wav.h` — WAV IO
- **Contract:** read PCM16/PCM24/float32 mono-or-stereo WAV into planar float,
  downmix/resample to mono, and write mono WAV as a file or to a
  non-seekable stdout stream.
- **Exact identifiers:** `WAV_S16 | WAV_S24 | WAV_F32` accepting exactly
  `wav16|wav24|wav32`; streaming header writes `data_size = 0x7FFFFFFFu` as an
  "unknown/live" marker; S16/S24 scale by `32767`/`8388607` **after** clamping,
  F32 is not clamped.
- **Gotcha:** `audio_mono_from_planar` **consumes and frees its input** on
  every path — passing a buffer you still need is a use-after-free.
- **Gotcha:** `wav.h` silently ignores channels 3+ while using `n_channels` for
  the stride — >2-channel files are mis-parsed, not rejected *inferred*.

### `src/audio-postproc.h`, `src/audio-postproc-stream.h` — post-processing
- **Contract:** strict port of `omnivoice/utils/audio.py` + `pydub.silence`:
  `remove_silence`, `cross_fade_chunks`, `peak_normalize_half`, `fade_and_pad`,
  `ref_preprocess_audio`, plus their streaming equivalents chained
  crossfader → silence remover → fade/pad.
- **Invariant:** all range arithmetic is done on **int16** to match pydub
  bit-for-bit: `f32→s16` is `(x * 32768).clip(-32768, 32767)` truncated toward
  0. This is deliberately *different* from the write path's `*32767` — do not
  unify them.
- **Invariant:** `ref_preprocess_audio` returns the **original** RMS (or
  `-1.0f` on empty); that value drives the loudness branch downstream:
  `< 0` → peak/0.5, `< 0.1` → rescale by `rms/0.1`, else no-op.
- **Gotcha:** the streaming chain's "volume scale" step is the **caller's**
  job — no struct implements it. Streaming voice-design output runs 6–12 dB
  below buffered because peak normalisation is skipped.
- **Gotcha:** streaming `flush` latency is up to `min_sil_n` samples (500 ms at
  24 kHz with mid=500); buffers grow monotonically over the whole stream.

### `src/philox.h` — PRNG
- **Contract:** Philox4x32-10 + Box-Muller reproducing PyTorch CUDA
  `torch.randn()/rand()` including bf16 rounding (`bf16_round` defaults true).
- **Invariant:** seed is the key, element index is the subsequence, offset 0;
  `ctr_lo` is a caller-advanced block counter. Same `(seed, subseq, offset)` →
  same floats on any platform.
- **Gotcha:** `philox_uniform_fill` doc says `[0,1)` but the formula yields
  `(0,1)` — the comment is wrong, the code is right.

### `src/rvq-file.h` — `.rvq` container
- **Contract (the format *is* the contract):** **no header, no magic.** Flat
  bitstream, `code_bits` bits per code, **LSB-first**, layout `[K, T]`
  row-major (all K codes of frame 0 first). `K` and `code_bits` come from the
  codec GGUF, never from the file. `T = (filesize * 8) / (K * code_bits)`.
- **Gotcha:** a wrong `K` or `code_bits` silently misparses as long as
  `n_codes % K == 0`. Write API takes no `K`, so the writer cannot validate
  grouping.

### `src/srt.h`, `src/utf8.h`, `src/debug.h`, `src/timer.h`, `src/ov-error.h`
- `srt.h`: tolerant SubRip parser → `SrtCue{index, t0, t1, text}` in seconds;
  accepts `.` or `,` ms separator, BOM, CRLF, missing index. **`srt_parse`
  always returns `true`** — treat `true` + empty vector as "no cues".
- `utf8.h`: Windows bridges UTF-8 ↔ UTF-16 at argv/`fopen`/console. Call
  `utf8_init(&argc, &argv)` as the **first line of `main()`**; `windows.h`
  must precede `shellapi.h`. `utf8_normalize` strips UTF-8 BOM everywhere and
  recodes UTF-16 BOMs.
- `debug.h`: dumps `[i32 ndims][dims...][f32 data]` to `<dir>/<name>.bin`;
  `debug_cosine_sim` returns `0.0` if either norm² < `1e-30`.
- `timer.h`: `ms()` does **not** sync the device — synchronize via a readback
  first or you measure queue submission.
- `ov-error.h`: `ov_set_error` (thread-local), `ov_throw` (`[[noreturn]]`,
  `std::runtime_error`), `ov_log`. Long messages are truncated, never split.
