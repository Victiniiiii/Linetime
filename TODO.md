# TODO

## Core

- [ ] Static ONNX Runtime build (eliminate .so dependency)
- [ ] Windows support (.exe build, cross-platform paths)
- [ ] macOS support (arm64 build)
- [ ] stdin pipe for lyrics text (avoid temp files)
- [ ] Batch mode (process multiple songs)
- [ ] Generate lyrics from audio

## Accuracy

- [ ] Test with more languages and genres
- [ ] Improve repeated-line handling (currently matched by order)
- [ ] Short-line merging (words split across lines by LLM)
- [ ] Confidence-based line filtering (skip low-confidence results)
- [ ] Adjustable CTC boost per-language

## Integration

- [ ] LRC validation / repair tool
- [ ] Export to other formats (SRT, VTT)

## Build

- [ ] Single-binary distribution (static ONNX Runtime)
- [ ] CI/CD pipeline
- [ ] Package managers (AUR, Homebrew)

## Nice to have

- [ ] GUI wrapper
- [ ] Real-time alignment preview
- [ ] Karaoke-style output
