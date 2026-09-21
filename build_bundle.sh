#!/bin/bash
set -e

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

BUILD_STATIC="build-static"
DIST_DIR="dist"
NPROC=$(nproc)
GPU_FLAG=""

echo "=== linetime bundle build ==="
echo ""

# Detect GPU provider
if [ -d "vendor/onnxruntime/lib" ] && ls vendor/onnxruntime/lib/*cuda* &>/dev/null 2>&1; then
    echo "  Detected CUDA ONNX Runtime"
    GPU_FLAG="-DORT_CUDA"
elif [ -d "vendor/onnxruntime/lib" ] && ls vendor/onnxruntime/lib/*coreml* &>/dev/null 2>&1; then
    echo "  Detected CoreML ONNX Runtime"
    GPU_FLAG="-DORT_COREML"
else
    echo "  Using CPU-only ONNX Runtime"
fi

# Step 1: Build whisper.cpp as static library (if not already)
if [ ! -f "$BUILD_STATIC/src/libwhisper.a" ]; then
    echo "[1/4] Building whisper.cpp (static)..."
    mkdir -p "$BUILD_STATIC"
    cd "$BUILD_STATIC"
    cmake ../vendor/whisper.cpp \
        -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF \
        -DGGML_OPENMP=OFF
    cmake --build . -j"$NPROC"
    cd "$SCRIPT_DIR"
else
    echo "[1/4] whisper.cpp static already built"
fi

echo "[2/4] Compiling linetime (fully static ONNX Runtime)..."
mkdir -p "$BUILD_STATIC/out"

WHISPER_SRC="vendor/whisper.cpp/include"
WHISPER_GGML_SRC="vendor/whisper.cpp/ggml/include"
GGML_STATIC="$BUILD_STATIC/ggml/src"
WHISPER_STATIC="$BUILD_STATIC/src"
ONNX_DIR="vendor/onnxruntime"

# Collect all ONNX Runtime static libs
ORT_LIBS=""
for lib in $(find "$ONNX_DIR/lib" -name "*.a" | sort); do
    ORT_LIBS="$ORT_LIBS -Wl,--whole-archive $lib -Wl,--no-whole-archive"
done

g++ -O3 -DNDEBUG -std=c++17 -Wno-unused-result \
    $GPU_FLAG \
    -Isrc -Ivendor -I"$WHISPER_SRC" -I"$WHISPER_GGML_SRC" -I"$ONNX_DIR/include/onnxruntime/core/session" \
    -o "$BUILD_STATIC/out/linetime" \
    src/main.cpp \
    src/audio.cpp \
    src/lyrics.cpp \
    src/ctc_aligner.cpp \
    src/whisper_aligner.cpp \
    src/merger.cpp \
    src/lrc_writer.cpp \
    "$WHISPER_STATIC/libwhisper.a" \
    "$GGML_STATIC/libggml.a" \
    "$GGML_STATIC/libggml-base.a" \
    "$GGML_STATIC/libggml-cpu.a" \
    $ORT_LIBS \
    -Wl,--allow-multiple-definition \
    -lpthread -ldl -lm -lgomp -lz

echo "[3/4] Packaging distribution..."
rm -rf "$DIST_DIR"
mkdir -p "$DIST_DIR/models"

# Binary
cp "$BUILD_STATIC/out/linetime" "$DIST_DIR/"

# ffmpeg
if [ -f "$SCRIPT_DIR/ffmpeg" ]; then
    cp "$SCRIPT_DIR/ffmpeg" "$DIST_DIR/"
elif command -v ffmpeg &>/dev/null; then
    cp "$(which ffmpeg)" "$DIST_DIR/"
    echo "  Bundled system ffmpeg: $(which ffmpeg)"
else
    echo "  WARNING: Place ffmpeg binary in $DIST_DIR/"
fi

# Models
for f in models/mms_multilingual.onnx models/mms_multilingual_tokenizer.json models/ggml-base.bin models/ggml-small.bin; do
    [ -f "$f" ] && cp "$f" "$DIST_DIR/models/"
done

echo "[4/4] Done!"
echo ""
echo "=== Distribution ==="
ls -lh "$DIST_DIR/"
echo ""
echo "=== Binary dependencies ==="
ldd "$DIST_DIR/linetime" 2>&1 | head -20
echo ""
echo "=== Usage ==="
echo "  cd $DIST_DIR && ./linetime song.wav lyrics.txt --method a --model-a models/mms_multilingual.onnx --tokenizer models/mms_multilingual_tokenizer.json"
echo "  GPU: ./linetime song.wav lyrics.txt --gpu"
