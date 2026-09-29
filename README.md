<div align="center">

# omnivoice.cpp

*Local AI text-to-speech with voice cloning, in C++17 on GGML*

[![Build (CUDA)](https://github.com/Aviteshmurmu19/omnivoice.cpp/actions/workflows/build-cuda.yml/badge.svg)](https://github.com/Aviteshmurmu19/omnivoice.cpp/actions/workflows/build-cuda.yml)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?style=flat-square&logo=nvidia&labelColor=gray)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

[Prebuilt binaries](#prebuilt-binaries) • [Quick start](#quick-start) • [Voice cloning](#voice-cloning) • [Building](#building-from-source) • [GPU support](#gpu-support)

</div>

A C++17 port of [OmniVoice](https://github.com/k2-fsa/OmniVoice) (Xiaomi / k2-fsa).
646 languages, 24 kHz mono output, voice cloning and voice design, running on
CPU, CUDA, ROCm, Metal and Vulkan.

This repository is a fork of
[`ServeurpersoCom/omnivoice.cpp`](https://github.com/ServeurpersoCom/omnivoice.cpp)
that adds **GitHub Actions builds for NVIDIA GPUs**, so you can get a working
binary without installing a compiler, CUDA or MSVC.

## Features

- **Voice cloning** — from a reference WAV plus its transcript
- **Voice design** — attribute keywords for gender, age, pitch, style, volume, emotion
- **Auto voice** — consistent speaker identity across long inputs
- **Long-form synthesis** — punctuation-aware chunking, voice-prompt promotion, cross-fade
- **Three tools** — `omnivoice-tts` (text to WAV), `omnivoice-codec` (WAV to RVQ codes) and `tts-server` (OpenAI-compatible HTTP)
- **Quantisation** — Q8_0 of the 612M-parameter Qwen3 backbone
- **Embeddable** — single-header, plain-C-linkage ABI for C, C++, Python ctypes, Rust bindgen and Go cgo

> [!NOTE]
> TPUs are not supported. ggml has no TPU or XLA backend, so `v6e-TPU` and
> `v5e-TPU` are out of reach for this project.

## Prebuilt binaries

CUDA builds for `sm_75` are published on the
[releases page](https://github.com/Aviteshmurmu19/omnivoice.cpp/releases):

| Asset | Size | Runs on |
|---|---|---|
| `omnivoice-windows-cuda-sm75.zip` | 564 MB | Windows x64 — GTX 1650, no CUDA toolkit needed |
| `omnivoice-linux-cuda-sm75.tar.gz` | 21 MB | Linux x64 — Colab and Kaggle T4 |

**The Windows archive is self-contained.** It ships `cudart64_12.dll`,
`cublas64_12.dll`, `cublasLt64_12.dll` and the MSVC runtime alongside the
executables, so an NVIDIA driver is the only requirement. Without those DLLs the
binaries fail at load time with `0xC0000135 STATUS_DLL_NOT_FOUND`, which users
see as `exit code -1073741515`.

> [!TIP]
> The archive is large because `cublasLt64_12.dll` alone is 643 MB. If you
> already have the CUDA runtime installed, skip the download and [build from
> source](#building-from-source) instead.

Linux builds do not bundle the CUDA runtime, because Colab and Kaggle images
already provide it.

Models are not included. Get them from
[Serveurperso/OmniVoice-GGUF](https://huggingface.co/Serveurperso/OmniVoice-GGUF):

```bash
huggingface-cli download Serveurperso/OmniVoice-GGUF \
  omnivoice-base-Q8_0.gguf omnivoice-tokenizer-Q8_0.gguf --local-dir models
```

## Quick start

```bash
echo "Hello world." | omnivoice-tts \
  --model models/omnivoice-base-Q8_0.gguf \
  --codec models/omnivoice-tokenizer-Q8_0.gguf \
  --lang English -o hello.wav
```

On Windows, pass the text on stdin as a UTF-8 file so the encoding is preserved:

```powershell
cmd /c "omnivoice-tts.exe --model models\omnivoice-base-Q8_0.gguf `
  --codec models\omnivoice-tokenizer-Q8_0.gguf --lang English -o hello.wav < prompt.txt"
```

## Voice cloning

Give it a reference clip and the matching transcript:

```bash
omnivoice-tts \
  --model models/omnivoice-base-Q8_0.gguf \
  --codec models/omnivoice-tokenizer-Q8_0.gguf \
  --ref-wav ref.wav --ref-text ref.txt \
  --lang Hindi -o out.wav < prompt.txt
```

The target language and the reference language do not have to match. The
reference transcript must be written to a file and passed with `--ref-text`.

### Pre-encoding the reference

`omnivoice-codec` encodes a reference WAV into a compact `.rvq` latent once, so
the encode is skipped on every later synthesis:

```bash
omnivoice-codec --model models/omnivoice-tokenizer-Q8_0.gguf -i ref.wav
omnivoice-tts \
  --model models/omnivoice-base-Q8_0.gguf \
  --codec models/omnivoice-tokenizer-Q8_0.gguf \
  --ref-rvq ref.rvq --ref-text ref.txt \
  --lang English -o out.wav < prompt.txt
```

> [!WARNING]
> An `.rvq` payload only decodes correctly against the same codec quantisation
> that produced it. Re-encode if you change `--codec`.

## HTTP server

`tts-server` exposes an OpenAI-compatible API with a registry of cloned voices
that stay resident in memory:

```bash
tts-server --model models/omnivoice-base-Q8_0.gguf \
           --codec models/omnivoice-tokenizer-Q8_0.gguf --port 8080
```

```bash
curl -X POST localhost:8080/v1/audio/voices -H "Content-Type: application/json" \
  -d "{\"name\":\"freeman\",\"ref_text\":\"$(cat ref.txt)\",\"rvq_b64\":\"$(base64 -w0 ref.rvq)\"}"

curl -X POST localhost:8080/v1/audio/speech -H "Content-Type: application/json" \
  -d '{"input":"Hello world.","voice":"freeman","language":"English","response_format":"wav"}' \
  -o out.wav
```

## Building from source

```bash
git clone --recurse-submodules https://github.com/Aviteshmurmu19/omnivoice.cpp
cd omnivoice.cpp
```

> [!IMPORTANT]
> `--recurse-submodules` is not optional. `ggml` is a submodule, and without it
> the build fails immediately at `add_subdirectory()` in `CMakeLists.txt`. A
> plain `git clone` is not enough — run `git submodule update --init --recursive`
> if you already cloned.

With a CUDA toolkit and MSVC in place:

```bash
cmake -S . -B build-msvc -G "Visual Studio 17 2022" -A x64 \
  -DGGML_CUDA=ON -DGGML_CUDA_FORCE_MMQ=ON -DGGML_BACKEND_DL=ON -DGGML_NATIVE=OFF
cmake --build build-msvc --config Release -j %NUMBER_OF_PROCESSORS%
```

Or use the scripts the project already ships: `buildcuda.sh`, `buildvulkan.sh`,
`buildcpu.sh`, `buildall.sh`, and the `.cmd` equivalents on Windows.

## GPU support

The CI builds one artifact per OS and architecture. Adding a GPU is a one-line
change to the `sm` matrix in [`.github/workflows/build-cuda.yml`](.github/workflows/build-cuda.yml):

| `sm` | Architecture | Hardware |
|---|---|---|
| `75` | Turing | GTX 1650, Tesla T4 |
| `80` | Ampere | A100 |
| `89` | Ada Lovelace | L4 |
| `120a` | Blackwell | RTX PRO 6000 (Colab G4) |

```yaml
matrix:
  os: [windows, linux]
  sm: ['75', '80']    # add architectures here
```

The toolkit is pinned to CUDA 12.8, the first release able to emit `sm_120`, so
Blackwell needs no toolkit change.

> [!TIP]
> `GGML_CUDA_FORCE_MMQ=ON` is enabled because Turing has no tensor cores, which
> covers both the GTX 1650 and the T4. Leave it off for Ampere and newer.

## Embedding the library

Single-header, single-name-prefix, plain C linkage, so C, C++, Python ctypes,
Rust bindgen and Go cgo all consume it the same way:

```c
#include "omnivoice.h"

struct ov_init_params iparams;
ov_init_default_params(&iparams);
iparams.model_path = "models/omnivoice-base-Q8_0.gguf";
iparams.codec_path = "models/omnivoice-tokenizer-Q8_0.gguf";

struct ov_context * ov = ov_init(&iparams);

struct ov_tts_params params;
ov_tts_default_params(&params);
params.text = "Hello world.";
params.lang = "English";

struct ov_audio audio = { 0 };
ov_synthesize(ov, &params, &audio);
/* audio.samples, audio.n_samples, audio.sample_rate, audio.channels */
ov_audio_free(&audio);
ov_free(ov);
```

Configure with `-DOMNIVOICE_SHARED=ON` for a shared library exporting only the
`ov_*` symbols. `tests/abi-c.c` is compiled with `-std=c99 -Wall -Werror
-pedantic` on every build, so a regression that breaks plain C consumability
fails the build rather than passing quietly.

## Design notes

Every non-obvious flag in the workflow is there for a reason, and the reasoning
is recorded with sources in
[`docs/superpowers/specs/2026-09-29-omnivoice-cuda-build-design.md`](docs/superpowers/specs/2026-09-29-omnivoice-cuda-build-design.md).
The short version:

- **`CMAKE_CUDA_ARCHITECTURES` is pinned.** The ggml default list starts
  `50-virtual 61-virtual 70-virtual`, which a CUDA 13.x toolkit cannot compile
  at all.
- **Windows uses `windows-2022`, not `windows-latest`.** Current images also ship
  Visual Studio 2026, and CMake picks the newest generator it finds, so an
  implicit generator resolves to one that has no CUDA toolset. Every project
  publishing Windows CUDA binaries pins `windows-2022` for the same reason.
- **CUDA is pinned to 12.8 rather than 13.x.** CUDA 13 removed offline
  compilation for Maxwell, Pascal and Volta, and PTX cannot JIT backwards, so a
  13.x build would not run on the Tesla P100 instances Kaggle and Colab gate on
  some tiers.

The two `ggml` repositories are easy to confuse: `ggml-org/ggml` is upstream and
is *not* what this project builds. The submodule points at
`ServeurpersoCom/ggml`, a tracking fork of upstream with no unique commits.

## Credits

Upstream model: OmniVoice by Xiaomi / k2-fsa, Apache 2.0.
Audio codec: Higgs Audio v2 (`bosonai/higgs-audio-v2-tokenizer`), Apache 2.0.
Original project: [`ServeurpersoCom/omnivoice.cpp`](https://github.com/ServeurpersoCom/omnivoice.cpp).
