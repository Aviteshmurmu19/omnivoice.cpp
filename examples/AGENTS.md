# AGENTS.md — `examples/`

Read [`../AGENTS.md`](../AGENTS.md) first (§3 CLI contracts, §8 run
commands). This file covers the example scripts and fixtures.

## Working directory differs per script — this is the main trap

| Script | Must be run from | Why |
|---|---|---|
| `tts.sh`, `clone.sh` | `examples/` | paths are `../build/omnivoice-tts`, `../models/...` |
| `tts.cmd`, `clone.cmd` | any (via Explorer/double-click) | prepends `%~dp0..\build\Release` to `PATH`, then `pause`s |
| `server.sh` | **repo root** | paths are `./build/tts-server`, `models/...` |
| `client.sh` | anywhere | takes `host`/`port` as `$1`/`$2` (defaults `127.0.0.1`, `8080`); needs `curl` **and** `ffplay` |

Running `server.sh` from `examples/`, or `tts.sh` from the root, fails with
"no such file" for the model/binary. Check cwd before "fixing" a path.

## Fixtures

- `prompt.txt` — the stdin text every TTS example pipes in. It is also the
  default `--prompt` of `tests/debug-tts-cossim.py`.
- `freeman.{wav,rvq,txt}` — reference triple for cloning: source audio,
  pre-encoded codes, transcript. `clone.sh` uses `--ref-rvq freeman.rvq
  --ref-text freeman.txt` (no `--ref-wav`, so the codec encode is skipped);
  `tests/debug-clone-cossim.py` uses `--ref-wav freeman.wav` instead. Both
  must keep working if you touch the encode or decode path.
- `freeman.wav` is tracked even though the root `.gitignore` has `*.wav`
  (it was added before/force-added) — regenerating or rewriting any `.wav`
  here will not show up in `git status` unless forced.

## Gotchas

- The scripts hard-code `omnivoice-tokenizer-Q8_0.gguf` as the codec while
  `models.sh` downloads `omnivoice-tokenizer-F32.gguf`; `server.sh` uses the
  F32 one. Any codec variant works — root §13 records the inconsistency.
- Example models are never in the repo (`models/*.gguf` is gitignored); a
  fresh clone cannot run these scripts until `models.sh` (or `convert.py` +
  `quantize.sh`) has been run.
- `.cmd` twins exist only for `tts` and `clone`; there is no Windows twin of
  `server.sh` / `client.sh`.
