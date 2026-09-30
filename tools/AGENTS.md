# AGENTS.md — `tools/`

Read [`../AGENTS.md`](../AGENTS.md) first: §3 holds the **flag contracts**
(what each CLI promises its user), §8 the build. This file covers the
*entry-point implementation* contract — what every `tools/*.cpp` must keep
doing when you add or change a flag.

## The four binaries

| Target | Links | API layer it uses | Error prefix |
|---|---|---|---|
| `omnivoice-tts` | `omnivoice-core` | `ov_*` ABI, **plus** `pipeline_*` directly in `--llm-test` / `--maskgit-test` (bypasses `ov_init`) | `[CLI]`, `[OmniVoice-TTS]` |
| `omnivoice-codec` | `omnivoice-core` | `pipeline_codec_*` directly, never `ov_*` | `[CLI]`, `[OmniVoice-Codec]` |
| `tts-server` | `omnivoice-core` + `httplib` + `yyjson` | `ov_*` ABI only | `[Server]` |
| `quantize` | **ggml only** — no `omnivoice-core` | ggml GGUF writer | `[Quantize]` |

`version.cmake` generates `version.h` into the build dir; every binary
includes it to print the `omnivoice.cpp <hash> (<date>)` banner via
`OMNIVOICE_VERSION`. Headers come from `src/` through CMake include dirs —
that is why `omnivoice-core` is a **static** library (tools need the
`pipeline_*` / `backend_*` symbols resolved directly).

## Contract shared by every tool

- **`utf8_init(&argc, &argv)` is the first statement of `main()`.** It sets
  the Windows console to UTF-8 and rebuilds argv from UTF-16; skip it and
  every non-ASCII flag value is mojibake on Windows.
- **Exit codes are only `0` (success) and `1` (anything else).** No distinct
  codes for usage vs runtime errors — scripts must not branch on them.
- **Failure goes through a top-level `try`/`catch` that prints
  `[<Tool>] FATAL: <what>` to stderr and returns 1.** `omnivoice-tts` and
  `omnivoice-codec` route through `main_impl()` for this; `tts-server` and
  `quantize` have no such catch — they rely on `ov_*` not throwing and on
  explicit `return 1` paths (*inferred*: an exception escaping either would
  `std::terminate`).
- **stdout carries audio/binary payloads only.** Everything else — usage,
  errors, `[CLI] Seed:`, progress — is stderr. `-o -` (tts) is the one
  payload channel on stdout; never `printf` diagnostics there.
- **`RVQ_CODE_BITS = 11` is defined independently in three files:**
  `omnivoice-tts.cpp`, `omnivoice-codec.cpp`, `tts-server.cpp`. `.rvq`
  files are headerless (see `src/rvq-file.h`), so a mismatch between the
  writers and readers corrupts silently. Change all three together.
- **Flag parsing is a hand-rolled `argv` loop with `i + 1 < argc` guards** —
  an unknown or truncated flag falls through to the usage message and exit 1.
  Validation runs after the whole loop, so flags are position-independent.

## Per-tool invariants

- **`omnivoice-tts`** — modes are mutually exclusive: TTS (default),
  `--llm-test`, `--maskgit-test`; `--ref-wav`/`--ref-rvq`, `--srt`,
  `--stream-by-line` are rejected outside synthesis mode. `--steps 0` is a
  *sentinel* meaning "library default 32", not a value the user may pass
  (`<1` is an error). `--seed -1` means `std::random_device`. Debug binary
  formats are documented in root §3.2 — they are a wire contract with
  `tests/debug-*-cossim.py`.
- **`omnivoice-codec`** — the output filename is derived by swapping the
  input extension; there is deliberately no output flag. Encode applies the
  same preprocessing as the TTS `--ref-wav` path (`ref_preprocess_audio`),
  which is what makes `clip.rvq` round-trip through `--ref-rvq` bit-identically
  — do not "optimise" either path independently.
- **`tts-server`** — owns no model state beyond one `ov_context`; synthesis
  serialises on `g_synth_mutex`, the voice registry on `g_voices_mutex`, and
  registered codes are **copied out under the lock** so a concurrent
  `DELETE`/`POST` cannot free memory a running synthesis still reads. Keep
  that copy. Request bodies are capped at 32 MB (`set_payload_max_length`).
- **`quantize`** — exactly `argc == 4`, no flags. Policy lives entirely in
  `should_quantize`; see root §3.5 for the never-quantise list. It writes
  metadata first, then appends tensor data, so a failed run leaves a partial
  output — and `quantize.sh` skips existing files, so that partial file is
  never retried *inferred*. Header says "Reads BF16 GGUF"; it accepts
  F32/F16/BF16 (divergence recorded in root §13).

## Gotchas when adding a flag

- Usage text and defaults drift: `--help` claims `--lang`/`--instruct`
  default to `'None'` while the code defaults them to `""` (root §13). If you
  add a flag, print the real default or fix both.
- `--dump <dir>` is parsed in `--llm-test` mode but never forwarded to that
  code path — parsing a flag that does nothing is worse than rejecting it.
- A flag that changes output *shape* (formats, batching, `T_audio` windows)
  must be checked against `tests/debug-*-cossim.py`, which hard-codes the
  `../build/omnivoice-tts` command line it expects.
