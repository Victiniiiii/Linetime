#!/bin/bash
# Build linetime (Method A + Method C)
set -e
cd "$(dirname "$0")"
g++ -O3 -DNDEBUG -std=c++17 -Wno-unused-result \
  -DORT_CUDA \
  -Isrc -Ivendor -Ivendor/onnxruntime/include/onnxruntime/core/session \
  -o linetime \
  src/main.cpp src/audio.cpp src/lyrics.cpp \
  src/ctc_aligner.cpp src/lrc_writer.cpp \
  src/transcriber.cpp src/reconcile.cpp \
  -Lvendor/onnxruntime/lib -lonnxruntime -lonnxruntime_providers_cuda -lonnxruntime_providers_shared \
  -Wl,-rpath,'$ORIGIN/lib' \
  -lpthread -ldl -lm
echo "built linetime"