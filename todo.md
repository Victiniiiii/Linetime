# TODO

- Batch mode (process multiple songs)
- Add four modes: Both lyrics and timestamps inputted (The script checks for correction and improves), only one inputted (The script takes that as a hint), none inputted (work from scratch)
- Return completion percentage in stdout (JSON progress output)
- Test with more languages and genres
- Short-line merging (words split across lines by LRC)
- Confidence-based line filtering (skip low-confidence results)
- Adjustable CTC boost per-language
- LRC validation / repair tool
- Export to other formats (SRT, VTT)
- Package managers (AUR, Homebrew)
- Do we need more than one method?
- Make this work with all main GPUs and auto detection (CUDA, CoreML, ROCm, etc.)
- Make this work for all languages and alphabets
- Fix Whisper GPU build (needs complete CUDA toolkit for nvcc; currently whisper.cpp CPU-only)
