# AGENTS.md — `.github/workflows/`

Instructions for editing these two workflows. Read
[`../../AGENTS.md`](../../AGENTS.md) first for the build traps; this file covers
how the workflows came to be, the invariants that keep them working, and the
documentation that must move with them.

## The two files

| File | Trigger | Does |
|---|---|---|
| `build-cuda.yml` | `workflow_dispatch` | Builds `{windows, linux} × {sm_75, sm_80, sm_89, sm_120a}` — 8 jobs, one artifact each |
| `publish-artifacts-to-release.yml` | `workflow_dispatch` | Moves a finished run's artifacts onto a release, server-side |

Neither triggers on push. A push is always safe.

Architectures: `75` Turing (GTX 1650, T4), `80` Ampere (A100), `89` Ada (L4),
`120a` Blackwell (RTX PRO 6000). Full table with artifact names is in the
README's *GPU support* section.

## How this got built

The original goal was to cross-compile for Windows inside Linux Colab, so no
build dependencies would be needed on Windows 11. That was abandoned:

- The `colab` CLI **cannot run on Windows at all**. `colab_cli/console.py` does a
  module-scope `import termios` (also `tty`, `pty`, `select`) and `cli.py` imports
  it transitively, so every subcommand dies at import. There are no
  `sys.platform`/`os.name` guards anywhere in the package — it is Linux/macOS only.
  The plan was to drive Colab *from* Windows, which is exactly where it fails.
- Cross-compiling CUDA for Windows from Linux needs the Windows toolkit plus an
  MSVC host compiler. Not realistic.
- GitHub's `windows-latest` already ships MSVC, so the pivot was plain Actions.

Then the failures, in order. Each one cost a full CI run:

1. **Linux: `apt` permission denied.** Without a container the job runs as the
   non-root `runner` user. Needs `sudo`.
2. **Windows: `No CUDA toolset found`.** `windows-latest` ships VS 2026, CMake
   takes the newest generator, CUDA 12.8 has no toolset for it. Fixed with
   `windows-2022` **and** an explicit generator.
3. **Windows: upload found no files after a 28m41s build.** The package step wrote
   `omnivoice-win-…` while the upload looked for `omnivoice-windows-…`.
4. **Windows: artifact shipped green but was incomplete.** The MSVC CRT glob used
   one `*` where the path needs two, and the only symptom was a warning buried in
   the CUDA install log.

The research that fixed #2 came from four projects that actually publish Windows
CUDA binaries — all pin `windows-2022`, none uses `windows-latest`. The research
that fixed #3 and #4 came from
[`niksedk/omnivoice.cpp`](https://github.com/niksedk/omnivoice.cpp), a fork of
this same project with a working Windows CUDA release, whose comments record the
`0xC0000135 STATUS_DLL_NOT_FOUND` failure mode in the wild.

> [!IMPORTANT]
> Upstream has **no** working CI. An identical `release.yml` in `acestep.cpp`,
> `minimaxmusic.cpp`, `yue2.cpp` and `s2s.cpp` has never executed and would fail
> anyway — no CUDA version pinned, so `Jimver/cuda-toolkit` defaults to 13.2.0,
> and no architecture pin, so it tries to compile Maxwell. Do not treat any of
> those as a template.

## Invariants

Break any of these and a build fails after ~30 minutes:

- **`runs-on` for Windows is `windows-2022`**, and the generator is pinned
  `-G "Visual Studio 17 2022" -A x64`. Both are load-bearing; the generator pin
  is what survives a future image change.
- **`CMAKE_CUDA_ARCHITECTURES` is always pinned**, from `matrix.sm.cuda`.
  ggml's default list starts `50-virtual 61-virtual 70-virtual`, uncompilable by
  CUDA 13.
- **`GGML_CUDA_FORCE_MMQ` is Turing-only, and is set conditionally.** It exists
  for GPUs with no tensor cores, which here means `sm_75` alone. On Ampere, Ada
  and Blackwell it would replace cuBLAS tensor-core GEMMs with slower integer dot
  products. If you add an architecture, check whether it has tensor cores before
  assuming the flag applies.
- **The `cuda` matrix axis stays on 12.x.** CUDA 13 dropped Maxwell/Pascal/Volta
  offline compilation, so a 13.x build cannot run on a Tesla P100, which Kaggle
  and Colab both offer. 12.8 is also the first toolkit that emits `sm_120`.
- **`sm` entries are `{cuda, name}` objects, not bare numbers.** Blackwell needs
  the `a` suffix: `cuda: '120a'` produces `120a-real`, and `name` is what appears
  in the artifact filename. A bare `'120'` would generate `120-real`.
- **Artifact filenames derive from `matrix.os` and `matrix.sm.name`** in both the
  package and upload steps. Never hardcode `win`, `linux` or a number.
- **`if-no-files-found: error` stays.** It is what caught failure #3.
- **A missing DLL throws; it never warns.** A green build that produced an
  artifact nobody can run is worse than a red one.
- **Windows bundles cuBLAS/cuDart; Linux does not.** Colab and Kaggle already
  ship the runtime, and the Windows DLLs are ~750 MB.

> [!WARNING]
> The `GGML_*` flags are **duplicated verbatim** in the Windows and Linux
> `Configure` steps, and can silently drift apart. Five are literal in both —
> `GGML_CUDA`, `GGML_BACKEND_DL`, `GGML_NATIVE`, `GGML_CUDA_CUB_3DOT2`,
> `CMAKE_CUDA_ARCHITECTURES` — with `CMAKE_BUILD_TYPE=Release` correctly
> Linux-only and `GGML_CUDA_FORCE_MMQ` handled separately as a conditional.
> Verify that rather than assume it.

> [!IMPORTANT]
> The PowerShell `Configure (Windows)` step builds its conditional flag as an
> **array** and splats it with `@mmq`. A bare `$mmq` that is an empty string
> still passes an empty argument to a native command, and CMake rejects it with
> `Unknown argument ""`. Bash is safe the other way — an unquoted empty variable
> expands to nothing — so the two steps deliberately differ in form.

## Keep documentation in sync

The flags and paths in these workflows are described in four places. A change to
one that is not reflected in the others is drift, and it is how the next agent
ends up "fixing" something deliberately pinned.

| Where | What it claims |
|---|---|
| `build-cuda.yml` inline comments | why each flag exists |
| `../../README.md` → *Design notes*, *GPU support* | pinned CUDA, `windows-2022`, the arch table |
| `../../AGENTS.md` → *CI traps* | the failure modes, for agents |
| `../../docs/superpowers/specs/2026-09-29-omnivoice-cuda-build-design.md` | the decision record, with sources |

When you change a flag, runner, architecture or artifact name, update **all
four** in the same commit. The spec is the source of truth and should record
*why*, including the project or document that established the fact. If you
cannot cite where a decision came from, either find the source or flag it as
unverified in the spec rather than stating it as fact.

Adding a GPU is a one-line change to the `sm` axis — but it is an **object**, and
Blackwell shows why:

```yaml
sm:
  - { cuda: '120a', name: '120a' }   # 120a, not 120
```

`cuda` becomes `-DCMAKE_CUDA_ARCHITECTURES=<cuda>-real`; `name` becomes the
artifact suffix. If you add an architecture, add its row to the README's *GPU
support* table in the same commit, and check whether it has tensor cores before
leaving MMQ alone.

## Verification

CI runners have no GPU. Green means it compiled, packaged and started — not that
it synthesises. The `--help` smoke test is `continue-on-error` and can pass with
missing DLLs, so a real WAV on hardware is the only genuine check.

Recorded baseline: GTX 1650, driver 610.74, Q8_0 weights, cross-language voice
clone, 4.96 s of audio, **RTF 4.4**, 23 s wall clock, `exit 0`. That figure is
from the build **before** `GGML_CUDA_FORCE_MMQ=ON` was added, so re-measure
before comparing anything against it.

Timing: Windows ~30 min, Linux ~15 min. Do not assume a hang.
