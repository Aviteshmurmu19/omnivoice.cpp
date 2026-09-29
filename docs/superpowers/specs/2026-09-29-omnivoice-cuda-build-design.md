# Design: CI CUDA build for omnivoice.cpp (Windows + Linux)

- **Date:** 2026-09-29
- **Status:** Draft for review
- **Target repo:** a fork of `ServeurpersoCom/omnivoice.cpp` (not yet created)

## Goal

Produce a Windows x64 CUDA build and a Linux x64 CUDA build of `omnivoice.cpp`
that a user can run **with no build dependencies installed locally**. The
Windows artifact in particular must run on a machine that has an NVIDIA driver
and nothing else.

Initial scope is `sm_75` only, covering the two GPUs available today.

## Non-goals

- Not cross-compiling. The original idea was to cross-compile from Linux Colab.
  Abandoned: the `colab` CLI cannot run on Windows (`colab_cli/console.py:20`
  imports `termios` at module scope, so every subcommand fails at import), and
  cross-compiling CUDA to Windows from Linux needs the Windows toolkit plus an
  MSVC host compiler. GitHub Actions runners supply a native Windows toolchain.
- Not publishing releases. `workflow_dispatch` only, artifacts only.
- Not supporting TPU. ggml has no TPU/XLA backend at the pinned submodule commit;
  `v6e-TPU` and `v5e-TPU` are out of reach for this project.
- Not bundling model weights. GGUFs come from `Serveurperso/OmniVoice-GGUF`.
- Not Vulkan, ROCm, SYCL, or Metal. NVIDIA CUDA only.

## Environment (measured, not assumed)

| Machine | GPU | VRAM | Driver | CUDA (UMD) | Arch |
|---|---|---|---|---|---|
| Windows 11 workstation | GTX 1650 | 4 GB | 610.74 | 13.3 | `sm_75` |
| Colab | Tesla T4 | 15 GB | 580.82.07 | 13.0 | `sm_75` |
| Kaggle | 2x Tesla T4 | 15 GB ea | 580.159.04 | 13.0 | `sm_75` |

Lowest driver across all three is 580.82.07, above the `>= 580` floor that
CUDA 13.x requires and far above the `>= 525` floor for 12.x.

The GTX 1650's 4 GB is the tightest resource. The 612M-parameter Q8_0 backbone is
roughly 700 MB, so it fits, but it is not spacious.

## Key decisions

### CUDA 12.8.1, pinned

`Jimver/cuda-toolkit` with `cuda: '12.8.1'` and `method: network`, identically on
both operating systems.

`Jimver/cuda-toolkit` is a thin wrapper over NVIDIA's own installers, so it has
no per-version capability list to go stale; it fails loudly and early if 12.8.1
is ever unavailable. Availability was confirmed via NVIDIA's
`redistrib_12.8.1.json`, which lists all required `windows-x86_64` components.

A `nvidia/cuda:12.8.1-devel-ubuntu24.04` container would be faster on Linux, and
the tag was verified to exist on Docker Hub, but selecting a container per-OS
requires a conditional `container:` expression and Windows runners reject
containers outright, so a mistake there fails both jobs before any build starts.
Uniform installation is worth a couple of minutes and removes an entire class of
failure. `method: network` is required on Linux and works on Windows; it also
avoids the full 12.x Windows installer, which bundles a display driver we do not
want installed on a GPU-less runner.

Why 12.x over 13.x, even though every measured driver clears 580:

- Kaggle and Colab both gate Tesla P100 (Pascal, `sm_60`) instances on some tiers.
  NVIDIA removed offline compilation for Maxwell, Pascal and Volta in CUDA 13.0,
  and PTX cannot JIT backwards, so a 13.x-built binary **cannot run on a P100 at
  all**. Since Kaggle is a deployment target, 12.x covers that case at no cost.
- 12.8 is also the first toolkit able to emit `sm_120`, so adding Blackwell later
  does not require a toolkit change.

`method: network` avoids running the full Windows installer. The CUDA 12.8 Windows
installer bundles a display driver; the 13.0+ one does not, but installing a
driver on a GPU-less CI runner is avoidable friction either way.

### `CMAKE_CUDA_ARCHITECTURES=75-real`, pinned explicitly

`-DCMAKE_CUDA_ARCHITECTURES=75-real`. Never rely on the ggml default. The default
list at the pinned submodule commit begins `50-virtual;61-virtual;70-virtual`,
which CUDA 13.x cannot compile at all. This pin is the single most important
defensive choice in the workflow.

`-real` rather than `-virtual` gives native SASS for the target instead of a PTX
JIT on every process start.

### One artifact per OS per architecture

A matrix, not a fat binary. Smaller artifacts, parallel jobs instead of one long
serial compile, and the user downloads only what is needed.

### Bundle the CUDA and MSVC runtime DLLs on Windows only

ggml's CMake is explicit that Windows has had no static cuBLAS since toolkit
12.3.1, so `cublas` must remain a DLL. An NVIDIA driver supplies `nvcuda.dll` but
**not** `cudart64_12.dll` or `cublas64_12.dll`; those ship only with the toolkit.
A non-bundled binary therefore fails to start on a machine without a CUDA
install, which would defeat the whole point.

Bundled into the Windows artifact, flattened beside the `.exe` so the standard
DLL search order finds them:

- `cudart64_12.dll`, `cublas64_12.dll`, `cublasLt64_12.dll` (from `$CUDA_PATH\bin`)
- `msvcp140.dll`, `vcruntime140.dll`, `vcruntime140_1.dll`, `concrt140.dll`
  (from the MSVC redist directory), ~0.8 MB total

The MSVC runtime DLLs are present on the current workstation and would not be
needed there, but they are bundled because a clean Windows 11 install is not
guaranteed to have them and the cost is negligible.

Linux artifacts do **not** bundle CUDA libraries. Colab and Kaggle images already
provide them, and the Windows-sized payload would be pure waste.

### Generator: MSBuild on Windows, Ninja on Linux

Follows the author's own sibling projects on Windows, where output lands in
`build-msvc/Release/` and MSBuild-specific log filtering is available. Ninja on
Linux matches llama.cpp. Not unified, deliberately: each is the proven choice for
its platform against this CMakeLists.

## Workflow structure

One new file, `.github/workflows/build-cuda.yml`. No changes to omnivoice.cpp
sources or to `CMakeLists.txt`.

```yaml
strategy:
  fail-fast: false
  matrix:
    os: [windows, linux]
    sm:  ["75"]     # adding a GPU later is a one-line change
runs-on: ${{ matrix.os == 'windows' && 'windows-latest' || 'ubuntu-24.04' }}
```

Adding a GPU later means adding one entry to `sm`: `80` (A100), `89` (L4),
`120a` (Blackwell / Colab G4). Nothing else changes.

The workflow is developed on a feature branch and merged into the fork's `master`
rather than committed straight to `master`. GitHub documents that
"`workflow_dispatch` ... only receives events when the workflow file is on the
default branch", so the "Run workflow" button does not appear until the workflow
has landed on `master`. Working through a branch also leaves `master` identical
to upstream, which keeps future `git pull upstream master` merges trivial and
makes the change reviewable as a diff.

### Common steps

1. `actions/checkout@v6` with **`submodules: recursive`**.
   Mandatory. The `ggml` submodule is the current hard blocker: the original
   clone used `--depth 1` without `--recurse-submodules`, so `ggml/` is empty and
   `CMakeLists.txt:66` `add_subdirectory(${GGML_SOURCE_DIR} ggml)` fails
   immediately on any platform. A plain `git clone` does not fix this either.
2. Install CUDA 12.8.1 via `Jimver/cuda-toolkit`, one shared step for both
   operating systems.
3. Configure and build with the flags above.
4. Smoke test: run `--help` on the four CLI tools (`omnivoice-tts`,
   `omnivoice-codec`, `tts-server`, `quantize`), `continue-on-error: true`.
   `test-abi-c` is also built but takes no useful `--help`, so it is left out.
5. Package and `actions/upload-artifact@v6`.

### Per-OS

**Windows** (`windows-latest`, `shell: pwsh`)

- `Jimver/cuda-toolkit@v0.2.36` with `cuda: '12.8.1'`, `method: network`.
  The input is named `cuda:`, not `cuda-version:`. The action caches the
  downloaded installer itself (`use-github-cache` and `use-local-cache` both
  default to `true`), so no separate `actions/cache` step is needed. The author's
  sibling projects add a manual cache over the same path, which is redundant.
- `cmake -S . -B build-msvc` with the shared flags, `--log-level=ERROR`.
- `cmake --build build-msvc --config Release -j $env:NUMBER_OF_PROCESSORS
  -- /v:minimal /consoleloggerparameters:ErrorsOnly`
- Copy `build-msvc/Release/*.exe` and `*.dll` to `dist/`, then the CUDA and MSVC
  runtime DLLs, then `Compress-Archive`.
- Artifact: `omnivoice-win-cuda-sm75.zip`

**Linux** (`ubuntu-24.04`)

- CUDA installed by the same `Jimver/cuda-toolkit` step as Windows
- `apt-get install -y cmake ninja-build build-essential git`
- `cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release` with the shared flags
- Copy `build/omnivoice-*`, `build/tts-server`, `build/quantize`, `build/*.so`
  to `dist/`, then `tar -czf`
- Artifact: `omnivoice-linux-cuda-sm75.tar.gz`

### Flags

```
-DGGML_CUDA=ON
-DGGML_BACKEND_DL=ON
-DGGML_NATIVE=OFF
-DCMAKE_CUDA_ARCHITECTURES=75-real
-DGGML_CUDA_CUB_3DOT2=ON
```

`GGML_BACKEND_DL=ON` makes the CUDA backend a separate `ggml-cuda.dll` loaded at
runtime by `ggml_backend_load_all()`, which is why that DLL must ship beside the
executables.

`GGML_CUDA_CUB_3DOT2=ON` pairs with the CCCL layout in the 12.x toolkit, matching
llama.cpp's tested 12.x recipe. It is omitted for 13.x, where CCCL is bundled
natively.

`GGML_CPU_ALL_VARIANTS=ON`, used by the author's sibling projects, is
deliberately omitted: it multiplies CPU backend DLLs and build time for no benefit
on an `sm_75` target. The default CPU backend still builds and acts as fallback.

## The two ggml repositories

There are two repos with similar names, and it matters which is which:

| Repo | Role |
|---|---|
| `ggml-org/ggml` | Upstream, canonical. Not built directly by this project. |
| `ServeurpersoCom/ggml` | The author's fork. **This is what omnivoice.cpp builds.** |

`omnivoice.cpp/.gitmodules` declares exactly one submodule, `ggml`, pointing at
`ServeurpersoCom/ggml.git`, pinned at `40e16e4a`.

The fork is a **clean tracking fork with no unique commits** of its own: `40e16e4a`
is present in `ggml-org/ggml`, and comparing the two shows identical results for
everything this design depends on — the default CUDA architecture list, the
Windows static-cuBLAS restriction, the `GGML_CUDA_CUB_3DOT2` option, and all 103
backend options. Neither repository contains a TPU or XLA backend.

This matters because the reference workflows in the Provenance section come from
`ggml-org/llama.cpp`, which is tested against `ggml-org/ggml`. Because the fork
tracks upstream exactly, that tested recipe applies to this build without
adaptation. Do not repoint `.gitmodules` at `ggml-org/ggml`; that would silently
change which code is compiled.

Upstream `master` was at `353b63b4` (v0.25.3) at the time of writing, slightly
ahead of the pinned `40e16e4a`. Advancing the pin is the author's call, not this
project's.

## Runner images: why not `windows-latest`

The first CI run failed on Windows with:

```
CMake Error at CMakeDetermineCompilerId.cmake:714 (message):
  No CUDA toolset found.
Call Stack:
  CMakeDetermineCUDACompiler.cmake:163
  ggml/src/ggml-cuda/CMakeLists.txt:59 (enable_language)
-- Building for: Visual Studio 18 2026
```

Current `windows-latest` images also ship Visual Studio 2026. CMake selects the
newest generator it discovers, so an implicit generator resolves to
`Visual Studio 18 2026`, for which CUDA 12.8 ships no MSBuild toolset. The fix is
`windows-2022` plus an explicit `Visual Studio 17 2022` generator, which together
guarantee the v143 toolset regardless of what a future image adds.

This is not a workaround invented here. Every project checked that publishes
Windows CUDA binaries pins `windows-2022`:

| Project | Runner | Generator | Bundles cuBLAS |
|---|---|---|---|
| `ggml-org/llama.cpp` | `windows-2022` | MSBuild, implicit | yes, 3 DLLs |
| `LostRuins/koboldcpp` | `windows-2022` | MSBuild | yes, 2 DLLs |
| `withcatai/node-llama-cpp` | `windows-2022` | — | — |
| `handy-computer/transcribe.cpp` | controlled 16-vCPU image | Ninja + `ilammy/msvc-dev-cmd` | no, consumer supplies it |

koboldcpp confirms the failure mode directly: it renames
`C:\Program Files\Microsoft Visual Studio\2022\Enterprise` to
`Enterprise_DISABLED` so CMake cannot select VS 2022 at all, then reinstalls an
older Visual Studio. That is the same conflict resolved with a hammer rather than
an image pin.

`handy-computer/transcribe.cpp` takes the other proven route, Ninja plus a
vcvars dev environment, on the reasoning that "nvcc needs cl.exe, and the Ninja
generator needs the MSVC environment". MSBuild was kept here instead because both
llama.cpp and koboldcpp build Windows CUDA with it, and the upstream author's own
`buildall.cmd` assumes an MSVC generator. Its install also uses
`Jimver/cuda-toolkit` with `method: network` and an explicit
`sub-packages` list, confirming that approach.

Two of the four bundle `cublas64_12.dll` and `cublasLt64_12.dll`; only
transcribe.cpp omits them, and it does so because it ships library bindings
consumed by package managers that supply their own runtime. A standalone CLI zip
meant to be unzipped and run is a different distribution shape, which is why the
CUDA DLLs are bundled here.

### Failure log from real runs

Three failures were hit and fixed. They are recorded because each was silent or
misleading in a way that would otherwise recur.

**1. Linux configure/install — missing `sudo`.** `apt-get` without a container
runs as the non-root `runner` user:
`E: Could not open lock file /var/lib/apt/lists/lock (13: Permission denied)`.

**2. Windows configure — `No CUDA toolset found`.** Covered under
"Runner images" above. Fixed by `windows-2022` plus an explicit generator.

**3. Windows upload — the whole job lost after a 29-minute build.** The package
step wrote `omnivoice-win-cuda-sm75.zip` while the upload step looked for
`omnivoice-windows-cuda-sm75.*`, because `win` was hardcoded in one place and
`matrix.os` expanded to `windows` in the other. Linux passed by luck, since its
name happened to match. Both archive names now derive from `matrix.os` so the
two cannot drift apart. The `if-no-files-found: error` guard is what caught it,
and it is kept for exactly this reason.

**4. Windows package — the MSVC runtime was silently missing.** The redist
layout is
`...\Microsoft Visual Studio\<year>\<edition>\VC\Redist\MSVC\<ver>\x64\Microsoft.VC<nnn>.CRT\`,
so the glob needs two path segments between `Visual Studio` and `\VC\`. A single
`*` matched nothing and only produced a warning line buried in the CUDA install
log, so the run went green while shipping an incomplete artifact. Now the DLLs
are globbed directly and a miss throws.

Lesson applied throughout: a packaging problem must fail the build. A green run
that produced an artifact nobody can run is worse than a red one, and the
missing-DLL symptom (`0xC0000135 STATUS_DLL_NOT_FOUND`, surfaced to users as
"exit code -1073741515") only appears at load time on a machine without a CUDA
toolkit, which is every end user.

## Verification

CI has no GPU, so it can only prove the binary compiles and starts. Validation is
split accordingly.

**In CI (automatic):** configure succeeds, build succeeds, all four binaries
respond to `--help`. This catches the submodule, toolkit, and arch-pin failures
that are the most likely ways to get stuck.

**On the GTX 1650 (manual):** download `omnivoice-win-cuda-sm75.zip`, extract,
and run with no CUDA Toolkit installed:

```
omnivoice-tts --model omnivoice-base-Q8_0.gguf --codec omnivoice-tokenizer-F32.gguf ^
  --lang English -o hello.wav
```

Confirm a non-empty WAV is produced. This is the real test of the zero-install
requirement; `--help` alone would pass even with missing DLLs in some cases, so
prefer a command that actually loads the model.

**On Colab and Kaggle T4 (manual):** upload the tarball, extract, run the same
command. Confirms the Linux artifact and the `sm_75` kernel actually execute.

## Risks and limitations

- **No GPU in CI.** A green build does not prove synthesis works. The manual steps
  above are mandatory, not optional.
- **CUDA 12.8 is pinned deliberately.** Bumping to 13.x is a one-line change but
  loses P100 coverage and requires removing `GGML_CUDA_CUB_3DOT2=ON`.
- **4 GB VRAM on the GTX 1650** is the most likely place a real run fails. The
  Q8_0 backbone fits, but there is little headroom for long-form synthesis.
- **Third-party actions.** `Jimver/cuda-toolkit` and `actions/*` are used by tag
  rather than by commit SHA. Pinning to SHAs would be more supply-chain-safe but
  is left as a known, accepted trade-off for a personal fork.

## Provenance of these decisions

Every non-obvious choice traces to something checked directly, not recalled:

- `ggml-org/llama.cpp` `.github/workflows/build-cuda-windows.yml`,
  `build-cuda-ubuntu.yml`, and the `windows-cuda` job in `release.yml` — the
  canonical proven Windows/Linux CUDA pipelines for this exact ggml codebase.
- `Jimver/cuda-toolkit` `action.yml` — the `cuda:` input name and its
  `13.2.0` default.
- NVIDIA CUDA 13.0 release notes — architecture removal, driver floors, and the
  Blackwell-support boundary.
- NVIDIA `redistrib_12.8.1.json` — Windows component availability.
- `ggml/src/ggml-cuda/CMakeLists.txt` and `ggml/CMakeLists.txt` at submodule
  `40e16e4a` (the SHA pinned by this fork's `master`) — the default arch list, the
  static-cuBLAS restriction, the `GGML_CUDA_CUB_3DOT2` option, and the backend
  inventory. Re-verified locally against the checked-out submodule after the
  initial research was done against an earlier SHA, `765bc96f`; all four
  conclusions were unchanged.
- The author's sibling projects `acestep.cpp`, `minimaxmusic.cpp`, `yue2.cpp`,
  `s2s.cpp` — the project's own idioms for checkout, CUDA install, MSBuild
  invocation, log filtering, and packaging.

**The author's own `release.yml` was deliberately not copied.** It is byte-for-byte
near-identical across four sibling projects but has never executed: `gh run list`
shows only the CPU-only `CI Build` workflow, and `acestep.cpp` has no releases. It
also passes no CUDA version, so it installs 13.2.0 and, with no architecture pin,
attempts to compile Maxwell, Pascal, and Volta targets that CUDA 13 removed. Its
Windows packaging also omits the CUDA runtime DLLs, which would leave the
resulting binary unable to start without a toolkit installed. The two deliberate
divergences, the arch pin and the DLL bundling, exist precisely to fix those two
defects.
