# Linetime

Turns an audio file and a lyrics file into a timed [LRC](https://en.wikipedia.org/wiki/LRC_(file_format)) file.

Give it a song and the words. It works out when each line is sung and writes an `.lrc` you can load into a karaoke player. If you don't have lyrics at all, it will transcribe the song for you.

Linux, macOS and Windows. CPU-only or CUDA GPU.

---

## Contents

- [Install](#install)
- [Quick start](#quick-start)
- [The three methods](#the-three-methods)
- [All options](#all-options)
- [Models](#models)
- [Building from source](#building-from-source)
- [How well does it work?](#how-well-does-it-work)
- [Known limits](#known-limits)
- [Requirements](#requirements)
- [License](#license)

---

## Install

Grab an archive from [the releases page](https://github.com/Victiniiiii/Linetime/releases) and unpack it. Each one bundles `linetime`, `whisper-cli`, `ffmpeg` and, on GPU builds, the CUDA libraries.

**You also need the models — about 4 GB, downloaded separately.** See [Models](#models).

| Archive | Platform |
|---|---|
| `linetime-linux-x64-cpu.tar.gz` | Linux, CPU |
| `linetime-linux-x64-gpu.tar.gz` | Linux, NVIDIA GPU |
| `linetime-windows-x64-cpu.tar.gz` | Windows, CPU |
| `linetime-windows-x64-gpu.zip` | Windows, NVIDIA GPU |
| `linetime-macos-universal.tar.gz` | macOS, Apple GPU (CoreML) |
| `linetime-whisper-cli-cuda-x64.tar.gz` | just the CUDA `whisper-cli`, to drop into an existing install |
| `linetime-whisper-cli-cuda-x64-windows.zip` | same, Windows |

Every archive needs its [models](#models) alongside it, in a `models/` directory next to the `linetime` binary. `./download_models.sh` fetches them.

## Quick start

You need `ffmpeg` (bundled in the release archives) and the models.

**You have the lyrics** — method C is usually what you want:

```bash
linetime song.mp3 lyrics.txt --method c --language en -o song.lrc
```

**You don't have the lyrics** — method B transcribes the song:

```bash
linetime song.mp3 --method b --language en -o song.lrc
```

**On a GPU**, add `--gpu`:

```bash
linetime song.mp3 lyrics.txt --method c --language en --gpu -o song.lrc
```

On Linux and macOS the bundled libraries need to be on the library path:

```bash
LD_LIBRARY_PATH=./lib ./linetime song.mp3 lyrics.txt --method c --language en --gpu
```

Lyrics can also come in on stdin, which is handy for piping:

```bash
cat lyrics.txt | linetime song.mp3 - --method c --language en
```

## The three methods

They differ in what they need from you and how much they trust the audio.

### Method C — Whisper + your lyrics *(best when you have lyrics)*

Whisper large-v3 listens to the song and transcribes it. Linetime then matches that against your lyrics and uses the result to:

- **fix typos** — if Whisper heard the same line you did but spelled a word differently, its version wins
- **fix structure** — if your lyrics have the wrong line breaks or a missing repeated chorus, the audio decides
- **re-time everything** — each line is then placed by MMS CTC forced alignment, so the timestamps come from the audio rather than from Whisper's guesses

Whisper's own timestamps are only kept where CTC can't align a line, such as a line it misheard badly.

### Method A — CTC only *(fastest, and needs no Whisper)*

Every line of your lyrics is force-aligned to the audio with the MMS multilingual model. Nothing is transcribed and nothing is invented — the text you supply is the text you get. It can't repair typos or restructure anything, because it never hears the words as words.

Use this when your lyrics are already correct and you just need timings.

### Method B — Whisper only *(no lyrics needed)*

Whisper transcribes the song and each segment becomes an LRC line. There is no lyric file to check against, so nothing can catch a mishearing — see [Known limits](#known-limits).

Whisper occasionally splits one lyric line into two segments, which would produce a junk line like `[01:01.80] dijela`. Linetime rejoins those automatically.

## All options

```
linetime <audio_file> [lyrics_file] [options]
linetime <audio_file> -            read lyrics from stdin
linetime <audio_file>              no lyrics, method b

  -o, --output <path>       Output LRC file (default: <audio>.lrc)
  --method <a|b|c>          Alignment method (default: a, or b without lyrics)
  --language <code>         Language hint for Whisper (default: auto)

  --gpu                     Use GPU acceleration (auto-detect CUDA/CoreML)
  --provider <name>         Force a provider: auto, cpu, cuda, coreml

  --model-a <path>          MMS CTC alignment model
  --model-c <path>          Whisper model for methods b and c
  --tokenizer <path>        Tokenizer JSON for the CTC model
  --whisper-cli <path>      Path to the whisper-cli binary
  --ffmpeg <path>           Path to the ffmpeg binary

  --lead <ms>               Shift all timestamps earlier by <ms>
  --boost <float>           CTC non-blank boost (default: 5.0)
  --min-confidence <float>  Drop lines below this alignment confidence (0-1)

  --recover-missing         Re-emit sung sections your lyrics omit
  --separate-vocals         Isolate the centre channel (default on for method b)
  --no-separate-vocals      Use the full mix instead
  --no-speech-threshold <0-1>
                            How confidently a window must look non-speech
                            before Whisper discards it (default: whisper's 0.6)

  --transcript <path>       Reuse a cached Whisper JSON transcript
  --verbose                 Print detailed alignment info
  -h, --help                Show this help
```

A few worth explaining:

**`--separate-vocals`** is on by default for method B. Most songs are stereo, and the two channels usually differ — vocals in the centre, instruments spread across the sides. Averaging to mono mixes them back together and Whisper struggles. This keeps the centre channel instead. Turn it off with `--no-separate-vocals` if the vocals genuinely aren't centred.

**`--recover-missing`** is for lyrics that are incomplete. If a chorus is sung four times but your lyrics only list it once, this finds the sung repetitions your file skipped and re-emits them.

**`--lead`** exists because different LRC files use different conventions for where a line starts. If your reference file consistently runs late, `--lead 200` will move everything 200 ms earlier.

**`--transcript`** replays a saved Whisper transcript instead of running Whisper again, which is what you want when iterating on alignment.

**`--provider cpu`** forces the CPU even on a machine with a GPU. Useful for checking whether a problem is GPU-related, or comparing the two.

## Models

Models live in a `models/` directory next to the `linetime` binary.

| File | Size | Needed for |
|---|---|---|
| `mms_multilingual.onnx` | 1.2 GB | methods A and C |
| `mms_multilingual_tokenizer.json` | 293 B | methods A and C |
| `ggml-large-v3.bin` | 3.0 GB | methods B and C |

Method A only needs the two small files. Methods B and C need Whisper too.

From a source checkout, `./download_models.sh` (needs `wget`) puts them in `models/` for you. Note it always writes next to *itself*, so if you're using an unpacked release archive, download them somewhere and move them into the archive directory yourself:

```bash
./download_models.sh              # creates ./models/
mv models <archive-dir>/models    # e.g. mv models ~/linetime-1.4.0/models
```

Or fetch the three files directly:

| File | URL |
|---|---|
| `mms_multilingual.onnx` | `https://huggingface.co/xenova/mms_multilingual/resolve/main/quantize.onnx` |
| `mms_multilingual_tokenizer.json` | `https://huggingface.co/xenova/mms_multilingual/resolve/main/tokenizer.json` |
| `ggml-large-v3.bin` | `https://huggingface.co/ggerganov/whisper.cpp/resolve/main/ggml-large-v3.bin` |

## Building from source

```bash
git clone --recurse-submodules https://github.com/Victiniiiii/Linetime.git
cd Linetime
./download_models.sh

cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DLINETIME_ONNXRUNTIME_STATIC=ON \
  -DLINETIME_BUILD_WHISPER_CLI=OFF
cmake --build build --target linetime --parallel $(nproc)
```

`LINETIME_ONNXRUNTIME_STATIC=ON` is worth keeping: the default links ONNX Runtime as a shared library, and the resulting binary won't start until you point `LD_LIBRARY_PATH` at it. The static link produces one self-contained executable. The CPU job in CI builds it this way too.

For a GPU build you need the shared ONNX Runtime package (static and CUDA are mutually exclusive) plus CUDA:

```bash
cmake -S . -B build -DCMAKE_BUILD_TYPE=Release \
  -DLINETIME_ONNXRUNTIME_DIR="$PWD/vendor/onnxruntime" \
  -DLINETIME_ONNXRUNTIME_STATIC=OFF \
  -DLINETIME_ENABLE_CUDA=ON \
  -DLINETIME_ENABLE_COREML=OFF \
  -DLINETIME_BUILD_WHISPER_CLI=OFF
cmake --build build --target linetime --parallel $(nproc)
```

Whisper is a separate program that `linetime` runs as a subprocess, not a library it links, so `--target linetime` leaves it out. A CPU build still needs `whisper-cli` at runtime for methods B and C; the release archives include it.

The binary lands in `build/bin/linetime` and looks for its models, libraries and helpers **relative to its own location**. So after a source build you need `models/` next to *that* binary, not at the repo root where `download_models.sh` puts it:

```bash
mkdir -p build/bin/models
cp models/mms_multilingual.onnx models/mms_multilingual_tokenizer.json \
   models/ggml-large-v3.bin build/bin/models/
cp dist/ffmpeg build/bin/          # or wherever your ffmpeg is
```

The usual finished layout:

```
linetime
models/
  mms_multilingual.onnx
  mms_multilingual_tokenizer.json
  ggml-large-v3.bin
lib/            # GPU builds only
bin/whisper-cli
ffmpeg
```

`build_bundle.sh` assembles that whole layout into `dist/` for you, detecting CUDA or CoreML automatically, which saves doing it by hand:

```bash
./build_bundle.sh                                  # auto-detect
LINETIME_ENABLE_CUDA=0 ./build_bundle.sh           # force CPU
```

To build `whisper-cli` yourself as well, set `-DLINETIME_BUILD_WHISPER_CLI=ON`. That needs a CUDA toolkit for a GPU build; point `CUDAToolkit_ROOT` and `CUDACXX` at one, or let the script use the vendored toolkit in `vendor/cuda-toolkit`. Compiling CUDA is memory-hungry, so the script sizes that step from your available RAM rather than your core count — override with `LINETIME_CUDA_JOBS`.

## How well does it work?

Measured against reference LRC files. **acc@1s** is the share of lines whose timestamp is within one second of the reference.

These are small test sets and reference LRCs are not always right, so read them as an indication of behaviour rather than a guarantee. Where a reference disagreed with the audio we checked it against [LRCLIB](https://lrclib.net) and, in one case, listened to the track.

**Method C** — 9 songs, 352 reference lines: **98.3%** within one second.

| | |
|---|---|
| Clean songs | acc@1s 1.00 |
| Worst case in the set | acc@1s 0.87 |

**Method A** — no transcription to get wrong, so its errors are placement, not words. On the one song in the test set where the reference itself is uncertain it scored 0.91.

**Method B** — 8 Bosnian/Serbian songs, 269 reference lines:

| | |
|---|---|
| Within 0.5 s | 0.50 |
| Within 1 s | 0.60 |
| Within 2 s | 0.65 |

Method B is transcription quality, not alignment quality, so its numbers are much lower. Treat it as a starting point to edit rather than a finished file.

## Known limits

Worth reading before you rely on method B.

**Method B cannot detect a mishearing.** It has no lyric file to check against, so if Whisper confidently invents a line over an instrumental passage, that line goes into the LRC. Linetime filters the two cases it can identify reliably — spoken broadcast narration ("thanks for watching") and stage directions ("[Music]") — and marks wordless vocal passages as `[humming]` so they stay easy to find and edit.

Repeating a real chorus is the case that remains. It's indistinguishable from a genuine repetition by any signal Linetime can measure: the audio is genuinely being sung, and Whisper is genuinely confident. **For karaoke, prefer method C** — you supply the words, so a hallucination has nothing to attach to.

**Method B lines are still coarse.** One line per Whisper segment, and a segment can span several real lyric lines. Linetime rejoins lines Whisper splits, but doesn't yet split lines that run long.

**Non-Latin scripts** partly defeat the hint matching in method C. A Russian or Korean lyric file won't match Whisper's Latin transliteration, and Linetime falls back to Whisper's own text.

**Output is anchored on Whisper's timing**, which is correct for the audio. If your reference LRC uses a different sync convention, `--lead` will reconcile it.

## Requirements

To run a release archive: nothing beyond `ffmpeg` and the models. `ffmpeg` is in the archive; the models are a separate ~4 GB download ([above](#models)).

To build from source:

- C++17 compiler
- CMake >= 3.14
- ffmpeg
- ONNX Runtime headers and libraries
- whisper.cpp (comes with `--recurse-submodules`)
- CUDA toolkit, only for GPU builds of `whisper-cli`

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).