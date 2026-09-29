<div align="center">

# omnivoice.cpp

*Local AI text-to-speech with voice cloning, in C++17 on GGML*

[![Build (CUDA)](https://github.com/Aviteshmurmu19/omnivoice.cpp/actions/workflows/build-cuda.yml/badge.svg)](https://github.com/Aviteshmurmu19/omnivoice.cpp/actions/workflows/build-cuda.yml)
[![CUDA 12.8](https://img.shields.io/badge/CUDA-12.8-76B900?style=flat-square&logo=nvidia&labelColor=gray)](https://developer.nvidia.com/cuda-toolkit)
[![License: MIT](https://img.shields.io/badge/License-MIT-blue?style=flat-square)](LICENSE)

[Prebuilt binaries](#prebuilt-binaries) • [Models](#models) • [Quick start](#quick-start) • [Voice cloning](#voice-cloning) • [Building](#building-from-source) • [GPU support](#gpu-support)

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

CUDA builds are published on the
[releases page](https://github.com/Aviteshmurmu19/omnivoice.cpp/releases), one per
OS and architecture — see [GPU support](#gpu-support) for the full list. The
Windows archives are around 564 MB each, the Linux ones about 21 MB.

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

Weights are not in the artifacts. See [Models](#models).

## Models

Pre-converted GGUFs live at
[`Serveurperso/OmniVoice-GGUF`](https://huggingface.co/Serveurperso/OmniVoice-GGUF).
You need two files: a **base** (the LLM) and a **tokenizer** (the audio codec).

| File | Size | Notes |
|---|---|---|
| [`omnivoice-base-Q8_0.gguf`](https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/omnivoice-base-Q8_0.gguf) | 626 MB | **Recommended.** Verified working; good quality at 4 GB VRAM |
| [`omnivoice-base-Q4_K_M.gguf`](https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/omnivoice-base-Q4_K_M.gguf) | 389 MB | Smaller, for tight VRAM |
| [`omnivoice-base-BF16.gguf`](https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/omnivoice-base-BF16.gguf) | 1174 MB | Reference precision |
| [`omnivoice-base-F32.gguf`](https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/omnivoice-base-F32.gguf) | 2342 MB | Unquantised |
| [`omnivoice-tokenizer-Q8_0.gguf`](https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/omnivoice-tokenizer-Q8_0.gguf) | 276 MB | **Recommended** |
| [`omnivoice-tokenizer-Q4_K_M.gguf`](https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/omnivoice-tokenizer-Q4_K_M.gguf) | 241 MB | Smaller |
| [`omnivoice-tokenizer-F32.gguf`](https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/omnivoice-tokenizer-F32.gguf) | 700 MB | Native dtype, what `quantize.sh` preserves |
| [`omnivoice-tokenizer-BF16.gguf`](https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/omnivoice-tokenizer-BF16.gguf) | 356 MB | — |

```bash
huggingface-cli download Serveurperso/OmniVoice-GGUF \
  omnivoice-base-Q8_0.gguf omnivoice-tokenizer-Q8_0.gguf --local-dir models
```

Or fetch one file directly:

```bash
curl -L -C - -o models/omnivoice-base-Q8_0.gguf \
  https://huggingface.co/Serveurperso/OmniVoice-GGUF/resolve/main/omnivoice-base-Q8_0.gguf
```

> [!NOTE]
> `-C -` matters. These files are 200 MB to 2.3 GB, and without it a dropped
> connection restarts the whole download.

To convert from the original checkpoint instead, use `checkpoints.sh` (fetch
`k2-fsa/OmniVoice`), then `convert.py` and `quantize.sh`.

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

CI builds one artifact per OS **and** per architecture — never a fat multi-arch
binary. Each is native SASS for its target, and you download only what you need.

| `sm` | Architecture | Hardware | Windows | Linux |
|---|---|---|---|---|
| `75` | Turing | GTX 1650, Tesla T4 | `omnivoice-windows-cuda-sm75.zip` | `omnivoice-linux-cuda-sm75.tar.gz` |
| `80` | Ampere | A100 | `omnivoice-windows-cuda-sm80.zip` | `omnivoice-linux-cuda-sm80.tar.gz` |
| `89` | Ada Lovelace | L4 | `omnivoice-windows-cuda-sm89.zip` | `omnivoice-linux-cuda-sm89.tar.gz` |
| `120a` | Blackwell | RTX PRO 6000 (Colab G4) | `omnivoice-windows-cuda-sm120a.zip` | `omnivoice-linux-cuda-sm120a.tar.gz` |

Adding one is a single line in [`.github/workflows/build-cuda.yml`](.github/workflows/build-cuda.yml):

```yaml
sm:
  - { cuda: '75',   name: '75' }
  - { cuda: '80',   name: '80' }
  - { cuda: '89',   name: '89' }
  - { cuda: '120a', name: '120a' }
```

`cuda` becomes the CMake architecture; `name` becomes the artifact name. The two
are separate because Blackwell needs the `a` suffix — `120a`, never `120`.

> [!TIP]
> The toolkit is pinned to CUDA 12.8, the first release able to emit `sm_120`, so
> Blackwell needs no toolkit change. It is also the last 12.x line, which is what
> keeps `sm_60` (Tesla P100) buildable.

> [!NOTE]
> `GGML_CUDA_FORCE_MMQ` is applied only to `sm_75`. It exists for GPUs with no
> tensor cores; forcing it on Ampere or newer would replace cuBLAS tensor-core
> GEMMs with slower integer dot products.

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
