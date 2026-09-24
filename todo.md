# TODO

- Batch mode (process multiple songs)
- Return completion percentage in stdout (JSON progress output)
- Test with more languages and genres
- Short-line merging (words split across lines by LRC)
- Confidence-based line filtering (skip low-confidence results)
- Adjustable CTC boost per-language
- LRC validation / repair tool
- Export to other formats (SRT, VTT)
- Package managers (AUR, Homebrew)
- Make this work for all languages and alphabets (non-Latin scripts: Korean, Russian)

## GPU Support
- [x] Linux CUDA GPU build (ONNX Runtime 1.19.2 + CUDA 12.6)
- [ ] Windows CUDA GPU build (CUDA in MSYS2 CI)
- [ ] macOS CoreML GPU build (needs CoreML ONNX Runtime)
- [ ] AMD ROCm support (via ONNX Runtime ROCm provider)