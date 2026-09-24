# Linetime

Automatic lyric-audio timestamp alignment. Takes an audio file and plain lyrics text, outputs [LRC](https://en.wikipedia.org/wiki/LRC_(file_format)) timed lyrics.

## How it works

1. Audio is decoded to 16kHz mono via ffmpeg
2. Lyrics text is parsed into individual lines
3. Alignment method:
   - **Method A (CTC):** MMS multilingual model forced-aligns each line to audio
   - **Method B (Whisper DTW):** Whisper base model aligns via dynamic time warping
   - **Method C (Whisper STT + correction):** Whisper large-v3 transcribes audio, then reconciles with your input lyrics — **fixing typos, organizing line/paragraph structure from audio, and correcting text** while using whisper timestamps
4. Results written as LRC with `[mm:ss.xx]` timestamps

## Quick start

```bash
# Clone and init submodule
git clone --recurse-submodules <repo-url>
cd linetime

# Download models (~1.9GB)
./download_models.sh

# Build
mkdir build && cd build
cmake .. -DCMAKE_BUILD_TYPE=Release
cmake --build . -j$(nproc)

# Run

**Method A (CTC forced alignment):**
```bash
./linetime song.wav lyrics.txt --method a \
  --model-a models/mms_multilingual.onnx \
  --tokenizer models/mms_multilingual_tokenizer.json
```

**Method B (Whisper DTW alignment):**
```bash
./linetime song.wav lyrics.txt --method b \
  --model-b models/ggml-base.bin \
  --language <lang_code>
```

**Method C (Whisper STT + hint correction):**
```bash
./linetime song.wav lyrics.txt --method c \
  --model-c models/ggml-large-v3.bin \
  --language <lang_code>
```
Method C transcribes audio with Whisper large-v3, reconciles with your lyrics (fixes typos, organizes structure from audio), and outputs timed LRC.

> **Note:** Method C works best for songs without repeated sections. Repeated choruses in the hint lyrics can cause timestamp drift, as the word-level DTW alignment may match repeats to the wrong audio occurrence. For songs with repeated structure, Method A (CTC) is more reliable.

## Usage

```
linetime <audio_file> <lyrics_file> [options]

Options:
  -o, --output <path>      Output LRC file (default: <audio>.lrc)
  --ffmpeg <path>          Path to ffmpeg binary (default: search PATH)
  --model-a <path>         MMS multilingual ONNX model
  --model-b <path>         Whisper GGML model
  --model-c <path>         Whisper large model for STT (default: models/ggml-large-v3.bin)
  --tokenizer <path>       Tokenizer JSON
  --language <code>        Language hint for whisper (default: auto)
  --method <a|b|c|both>    Alignment method (default: a)
  --verbose                Print detailed alignment info
  -h, --help               Show this help

Methods:
  a     CTC forced alignment only (MMS multilingual)
  b     Whisper DTW alignment only
  c     Whisper STT + hint reconciliation (re-derive structure, fix typos, organize lines/paragraphs; best for non-repeating lyrics)
  both  Hybrid: run both CTC and Whisper DTW, merge best results
```

## Bundle build

For a self-contained distribution with bundled dependencies:

```bash
./build_bundle.sh
# Output in dist/
```

This produces a `dist/` directory containing the binary, ONNX Runtime shared libraries, ffmpeg, and model files. Only system libraries (libc, libstdc++) are required at runtime.

## Models

| Model | Size | Purpose |
|-------|------|---------|
| `mms_multilingual.onnx` | ~1.2GB | CTC forced alignment (Meta MMS) |
| `mms_multilingual_tokenizer.json` | 293B | Tokenizer for CTC model |
| `ggml-base.bin` | ~142MB | Whisper base model |
| `ggml-small.bin` | ~466MB | Whisper small model (better accuracy) |
| `ggml-large-v3.bin` | ~3.1GB | Whisper large-v3 (STT + hint reconciliation, method c) |

Run `./download_models.sh` to fetch all models.

## Requirements

- C++17 compiler
- CMake >= 3.14
- ffmpeg (for audio decoding)
- ONNX Runtime (for CTC inference)
- whisper.cpp (built as submodule)

## License

GNU General Public License v3.0, see [LICENSE](LICENSE).
