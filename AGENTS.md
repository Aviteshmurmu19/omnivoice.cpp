# AGENTS.md

Fork of [`ServeurpersoCom/omnivoice.cpp`](https://github.com/ServeurpersoCom/omnivoice.cpp)
that adds GitHub Actions CUDA builds. Read this before touching the workflow — the
build traps below each cost a full 30-minute run to diagnose.

Design rationale and sources: `docs/superpowers/specs/2026-09-29-omnivoice-cuda-build-design.md`.
Docs belong in that directory, in this repo.

## Setup

The `ggml` submodule is mandatory. A plain `git clone` leaves it empty and the
build dies at `add_subdirectory(${GGML_SOURCE_DIR} ggml)` in `CMakeLists.txt`.

```bash
git clone --recurse-submodules <url>
git submodule update --init --recursive   # if already cloned
```

A leading `-` in `git submodule status` means uninitialised and the build will
fail. Check this first when a build breaks for no visible reason.

Remotes: `origin` is this fork, `upstream` is the original. Update with
`git pull upstream master`.

> [!IMPORTANT]
> There are two ggml repositories and they are not interchangeable.
> `ggml-org/ggml` is upstream. This project builds `ServeurpersoCom/ggml`, a
> tracking fork whose pinned commit exists verbatim in upstream and which carries
> no unique commits. Do not repoint `.gitmodules` at `ggml-org/ggml` — that
> silently changes which code is compiled.

## Build

`CMakeLists.txt` sets `CMAKE_RUNTIME_OUTPUT_DIRECTORY` with a plain `set()`, so
`-DCMAKE_RUNTIME_OUTPUT_DIRECTORY=...` is **ignored**. Output lands in:

- Windows, MSVC multi-config: `build-msvc/Release/`
- Linux, Ninja single-config: `build/`

`tools/version.cmake` only runs `git rev-parse --short HEAD` and
`git show -s HEAD`, both valid at `fetch-depth: 1`, with an `"unknown"` fallback.
`fetch-depth: 0` is not needed.

## CI traps

**Never use `windows-latest` for a CUDA build.** It ships Visual Studio 2026 and
CMake selects the newest generator it finds, so an implicit generator resolves to
`Visual Studio 18 2026` and configure fails with `No CUDA toolset found` at
`enable_language(CUDA)`. Use `windows-2022` **and** pin
`-G "Visual Studio 17 2022" -A x64`. Every project that publishes Windows CUDA
binaries pins `windows-2022`; koboldcpp works around it by renaming the VS 2022
directory so CMake cannot select it.

**Always pin `CMAKE_CUDA_ARCHITECTURES`.** ggml's default list starts
`50-virtual 61-virtual 70-virtual`, which a CUDA 13.x toolkit cannot compile at
all. Unpinned means a broken build the moment the toolkit moves.

**Stay on CUDA 12.x.** CUDA 13 removed offline compilation for Maxwell, Pascal
and Volta, and PTX cannot JIT backwards, so a 13.x build cannot run on a Tesla
P100 (`sm_60`) — which Kaggle and Colab both gate on some tiers. 12.8 is also
the first toolkit that can emit `sm_120`, so Blackwell needs no toolkit change.

**Bundle the CUDA DLLs on Windows.** cuBLAS has no static library on Windows
since toolkit 12.3.1, and a driver supplies `nvcuda.dll` but neither cuBLAS nor
cuDart. Ship `cudart64_*`, `cublas64_*` and `cublasLt64_*` next to the
executables or they fail at load with `0xC0000135 STATUS_DLL_NOT_FOUND`
(surfaced as `exit code -1073741515`). Glob the versions rather than hardcoding
`_12`, and also probe `bin\x64` — CUDA 13 moved them there.

**The MSVC CRT glob needs two path segments** after `Visual Studio`:

```
$env:ProgramFiles\Microsoft Visual Studio\*\*\VC\Redist\MSVC\*\x64\Microsoft.VC*.CRT\*.dll
```

One `*` silently matches nothing. The artifact shipped green and broken with
only a warning buried in the CUDA install log.

**Derive artifact names from `matrix.os`, never a literal.** A hardcoded `win` in
the package step against `matrix.os` = `windows` in the upload step discarded a
29-minute build at the last step. Keep `if-no-files-found: error` — it is what
caught that.

**`gh run download` inside a workflow needs `--repo`.** The workflow has no
checkout, so `gh` has no git context to infer the repository from.

**Release notes with backticks:** write them to a file and use
`gh release edit --notes-file`. A PowerShell here-string eats the backticks and
leaves `\foo\`.

**`GGML_CUDA_FORCE_MMQ` is applied to `sm_75` only.** The GTX 1650 and the T4 are
Turing and have no tensor cores, so ggml falls back on its own and the flag just
makes that explicit. On Ampere, Ada and Blackwell it would replace cuBLAS
tensor-core GEMMs with slower integer dot products, so the workflow sets it
conditionally rather than globally. If you add an architecture, check whether it
has tensor cores before assuming the flag applies.

Budget the time: Windows ~30 min, Linux ~15 min, and the four architectures run
as parallel jobs. The Windows artifact is 564 MB, almost entirely
`cublasLt64_12.dll` (643 MB uncompressed).

## Do not copy upstream's CI

Upstream has no working CI anywhere, including here (`/.github/workflows` 404s).
An identical `release.yml` in `acestep.cpp`, `minimaxmusic.cpp`, `yue2.cpp` and
`s2s.cpp` has **never executed** — and would fail anyway: it passes no CUDA
version, so `Jimver/cuda-toolkit` defaults to 13.2.0, and with no architecture
pin it tries to compile Maxwell.

Proven references, in order of usefulness: `niksedk/omnivoice.cpp` (same
project, working Windows CUDA release), `handy-computer/transcribe.cpp` (ggml,
Ninja + `ilammy/msvc-dev-cmd`, lean `sub-packages`), `ggml-org/llama.cpp`,
`LostRuins/koboldcpp`, `withcatai/node-llama-cpp`.

## Runtime

`omnivoice-tts` takes `--ref-text` as a **file path**, not inline text; the target
text is read from **stdin**, and `-o -` streams to stdout. `--lang` matches
case-insensitively against a table keyed lowercase (`hindi`).

Verified end to end on a GTX 1650 with Q8_0 weights: cross-language voice
clone, 4.96 s of audio, RTF 4.4, 23 s wall clock, `exit 0`. That RTF is from
the released build, which predates `GGML_CUDA_FORCE_MMQ=ON` — re-measure before
quoting it against a newer build.

No TPU support — ggml has no TPU or XLA backend.

## GPU architectures

`build-cuda.yml` builds `{windows, linux} x {sm_75, sm_80, sm_89, sm_120a}` — eight
jobs, one artifact each, `fail-fast: false` so one bad architecture cannot block
the rest. Entries are `{cuda, name}` objects, not bare numbers: Blackwell needs
the `a` suffix, so `cuda: '120a'` yields `120a-real` and a bare `'120'` would not.
`name` is what appears in the artifact filename.

`sm_60` is deliberately absent. It is buildable on CUDA 12.x and is the reason the
toolkit is not on 13.x, but no current runner GPU here is Pascal. Add it as
`{ cuda: '60', name: '60' }` if a Tesla P100 becomes a target.

## Verification

CI runners have no GPU. A green run proves the build compiles, packages and
starts; it does **not** prove synthesis. The `--help` smoke test is a
`continue-on-error` step and can pass with missing DLLs. Always confirm a real
WAV comes out on hardware.

Models are not in the repo (`models/*.gguf` is gitignored) and not in the
artifacts. Fetch from
[`Serveurperso/OmniVoice-GGUF`](https://huggingface.co/Serveurperso/OmniVoice-GGUF).

## Gotcha

This fork's `README.md` replaces upstream's, so `git pull upstream master`
conflicts on `README.md` whenever upstream edits theirs. Resolve by keeping this
repo's version, and check whether upstream's README gained anything worth
porting.
