# TODO

- Batch mode (process multiple songs)
- Add four modes: Both lyrics and timestamps inputted (The script checks for correction and improves), only one inputted (The script takes that as a hint), none inputted (work from scratch)
- Fix GPU acceleration mostly not recognizing the GPU
- Return completion percentage in stdout (JSON progress output)
- Test with more languages and genres
- Improve repeated-line handling (currently matched by order)
- Short-line merging (words split across lines by LLM)
- Confidence-based line filtering (skip low-confidence results)
- Adjustable CTC boost per-language
- LRC validation / repair tool
- Export to other formats (SRT, VTT)
- Package managers (AUR, Homebrew)
