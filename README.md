# Linetime

Automatic lyric-audio timestamp alignment. Takes an audio file and plain lyrics text, outputs [LRC](https://en.wikipedia.org/wiki/LRC_(file_format)) timed lyrics.

## How it works

1. Audio is decoded to 16kHz mono via ffmpeg
2. Lyrics text is parsed into individual lines
3. CTC forced alignment (MMS multilingual model) maps each line to a time range in the audio
4. Results are written as an LRC file with `[mm:ss.xx]` timestamps

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
./linetime song.wav lyrics.txt --method a \
  --model-a ../models/mms_multilingual.onnx \
  --tokenizer ../models/mms_multilingual_tokenizer.json
```

## Usage

```
linetime <audio_file> <lyrics_file> [options]

Options:
  -o, --output <path>      Output LRC file (default: <audio>.lrc)
  --model-a <path>         MMS multilingual ONNX model
  --model-b <path>         Whisper GGML model
  --tokenizer <path>       Tokenizer JSON
  --language <code>        Language hint for whisper (default: auto)
  --method <a|b|both>      Alignment method (default: a)
  --boost <float>          CTC non-blank boost (default: 5.0)
  --verbose                Print detailed alignment info
  -h, --help               Show this help

Methods:
  a     CTC forced alignment only (MMS multilingual)
  b     Whisper DTW alignment only
  both  Hybrid: run both, merge best results
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

Run `./download_models.sh` to fetch all models.

## Project structure

```
linetime/
├── CMakeLists.txt          # Main build (shared libs, ffmpeg)
├── CMakeLists.static.txt   # Static whisper + bundled ONNX build
├── build_bundle.sh         # Bundle distribution builder
├── download_models.sh      # Model downloader
├── src/
│   ├── main.cpp            # CLI entry point
│   ├── audio.cpp/h         # Audio loading (via ffmpeg)
│   ├── ctc_aligner.cpp/h   # CTC forced alignment (ONNX Runtime)
│   ├── whisper_aligner.cpp/h  # Whisper DTW alignment
│   ├── lyrics.cpp/h        # Lyrics text parser
│   ├── merger.cpp/h        # CTC + Whisper result merger
│   ├── lrc_writer.cpp/h    # LRC file writer
│   └── utils.h             # Math utilities
└── vendor/
    ├── json.hpp            # nlohmann/json
    └── whisper.cpp/        # whisper.cpp (git submodule)
```

## Requirements

- C++17 compiler
- CMake >= 3.14
- ffmpeg (for audio decoding)
- ONNX Runtime (for CTC inference)
- whisper.cpp (built as submodule)

## License

GNU General Public License v3.0 — see [LICENSE](LICENSE).
