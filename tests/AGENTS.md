# AGENTS.md — `tests/`

Read [`../AGENTS.md`](../AGENTS.md) first for the project-wide contracts. This
file covers the parity harnesses, the committed baseline logs, and the C ABI
gate that lives here.

## What is in here

| File | Role |
|---|---|
| `abi-c.c` | C99 ABI gate. Built by the **root** CMake as target `test-abi-c`, on every build, `-Wall -Werror -pedantic`. Never loads a model. |
| `debug-tts-cossim.py` | Voice-**design** path: runs `../build/omnivoice-tts` and the PyTorch reference on the same input, dumps every stage to `cpp/` + `python/`, prints cosine similarity per stage. |
| `debug-clone-cossim.py` | Voice-**clone** path: same comparison including `RefAudioCodes` and the per-layer hidden states. |
| `debug-*.sh` | Loops `GGML_BACKEND` ∈ `CUDA0 Vulkan0 CPU` × `--quant` ∈ `F32 BF16 Q8_0 Q4_K_M`, `tee`-ing into the `*.log` baselines. |
| `cross-decode.py` | Cross check: feed GGML-produced tokens to the PyTorch codec and vice versa, to localise which side of a divergence lives where. |
| `tts-*.log`, `clone-*.log` | 24 committed baseline runs (2 harnesses × 3 backends × 4 quants). |

## Running the harnesses

```bash
cd tests                                  # cwd matters: all paths are relative
python debug-tts-cossim.py --quant F32    # or --quant Q8_0 ...
GGML_BACKEND=CUDA0 python debug-clone-cossim.py --quant Q8_0
./debug-tts-cossim.sh                     # full 12-run matrix, tees to tts-<backend>-<quant>.log
```

- **Build first:** `BIN = "../build/omnivoice-tts"` is hard-coded; on Windows
  the binary lands in `build-msvc/Release/`, so this harness is Linux/GPU-lab
  shaped, not CI shaped.
- **Python deps:** `numpy`, `soundfile`, `torch`, `torchaudio`, and the
  upstream `omnivoice` package installed (the reference implementation). It is
  **not** in this repo — `pip install` it from `k2-fsa/OmniVoice`.
- Defaults: `--prompt ../examples/prompt.txt`, clone adds
  `--ref-text ../examples/freeman.txt --ref-wav ../examples/freeman.wav`,
  `--seed 42`, `--lang English`. The clone harness emits `--lang French` in the
  committed baselines (see the `[Input] Language:` line), so reruns with the
  current default will not reproduce them exactly.
- Dumps land in `tests/cpp/` and `tests/python/` — both are directories, which
  is exactly what the root `.gitignore` rule `tests/*/` exists to ignore.
- Exit code is mechanical only: `sys.exit(1)` when a model/binary is missing,
  otherwise the child's return code. **No cosine-similarity threshold is
  asserted anywhere** — a run that diverges completely still exits 0.

## What the baseline logs actually show

Read `Tokens:` / `Audio:` / `WAV stft_cos` at the tail of each log. Current
state of the committed runs:

| Path | Result in the committed logs |
|---|---|
| `tts-CPU-F32`, `tts-CUDA0-F32` | Full parity: `Tokens exact: 100.00%`, `Audio: 0.99999x`, `stft_cos: 1.000000` |
| `tts-*` at BF16 / Q8_0 / Q4_K_M, and `tts-Vulkan0-F32` | Diverges: `Tokens exact` 0.3–27%, `Audio` 0.008–0.59. Early stages still agree (prompt ids 100%, logits cos ≈ 1.0); the split starts at step-0 argmax ties and MaskGIT amplifies it. |
| **every** `clone-*` log, including F32 | Partial only: `RefAudioCodes exact` 97–98%, `Step0 pred_tokens` 94–96%, final `Tokens exact` ≤ 19.96%, `Audio` ≤ 0.28. Divergence begins in the reference-encoder codes, before the LM. |

Interpretation rules:

- These logs are **evidence, not fixtures** — nothing diffs or asserts on
  them, and they contain machine-dependent text (tqdm `it/s`, timings), so a
  byte diff against a rerun is meaningless. Compare the `[Cossim]` lines only.
- A green harness run is **not** a parity pass. Check the numbers.
- If you change numerics (dtype, reduction order, RoPE, RNG), rerun at least
  `debug-tts-cossim.py --quant F32` and confirm it still ends
  `Tokens exact: 100.00%` — that is the one configuration here where parity is
  currently demonstrated end to end.

## `abi-c.c` — the ABI gate

Built by default at the repo root, so it fails the main build rather than an
opt-in step. Exit codes (the contract, verbatim from the file):

```
0 pass          5 ov_last_error() empty after a known failure
1 defaults/abi  6 log callback never invoked
2 ov_init(NULL) non-NULL   7 last log level != OV_LOG_ERROR
3 ov_synthesize(NULL) wrong status
4 ov_duration_sec_to_tokens(NULL,...) < 1
8 future abi_version accepted   9 ov_extract_voice_ref(NULL,...) wrong status
```

It asserts `mg_num_step == 32`, `chunk_duration_sec > 0`, that both
`*_default_params` stamp `abi_version = OV_ABI_VERSION`, and that a `NULL`
handle yields `OV_STATUS_INVALID_PARAMS`. When you add a public entry or a
field whose default is contractual, extend this file in the same commit.

## Gotchas

- `*.wav` is globally gitignored, and the harnesses write WAVs into the
  already-ignored `cpp/` and `python/` dirs — don't expect audio artifacts to
  be committable.
- `docs/ARCHITECTURE.md` §Validation quotes `Tokens 100% / Audio 0.9999` for
  **both** the TTS and Clone paths; only the TTS F32 logs match that today
  (see the table above). Don't quote the clone figure from the doc without
  re-running it.
- `docs/ARCHITECTURE.md` §Module map lists `tests/prompt.txt`,
  `tests/ref-audio.wav`, `tests/ref-text.txt` — those files do not exist; the
  real fixtures are `examples/prompt.txt` and `examples/freeman.{wav,txt}`.
