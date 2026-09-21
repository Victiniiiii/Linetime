#!/bin/bash
set -e

# Build ONNX Runtime with GPU support
# Usage:
#   ./build_ort_gpu.sh cuda     # NVIDIA CUDA (Linux/Windows)
#   ./build_ort_gpu.sh coreml   # Apple CoreML (macOS)
#   ./build_ort_gpu.sh all      # All available providers

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ORT_VERSION="1.19.2"
BUILD_DIR="$SCRIPT_DIR/build-ort-gpu"
INSTALL_DIR="$SCRIPT_DIR/vendor/onnxruntime"

PROVIDER="${1:-auto}"

echo "=== Building ONNX Runtime $ORT_VERSION with GPU support ==="
echo "Provider: $PROVIDER"

# Clone ONNX Runtime if not already present
if [ ! -d "$BUILD_DIR/onnxruntime" ]; then
    echo "[1/5] Cloning ONNX Runtime..."
    mkdir -p "$BUILD_DIR"
    git clone --recursive --branch v$ORT_VERSION --depth 1 \
        https://github.com/microsoft/onnxruntime.git \
        "$BUILD_DIR/onnxruntime"
fi

cd "$BUILD_DIR/onnxruntime"

# Build flags
CMAKE_ARGS="-DCMAKE_BUILD_TYPE=Release -DCMAKE_INSTALL_PREFIX=$INSTALL_DIR"

case "$PROVIDER" in
    cuda)
        echo "[2/5] Configuring for CUDA..."
        CMAKE_ARGS="$CMAKE_ARGS \
            -DONNXRUNTIME_USE_CUDA=ON \
            -DCUDNN_HOME=/usr/local/cuda \
            -DCUDA_HOME=/usr/local/cuda"
        BUILD_FLAG="--use_cuda"
        ;;
    coreml)
        echo "[2/5] Configuring for CoreML..."
        CMAKE_ARGS="$CMAKE_ARGS \
            -DONNXRUNTIME_USE_COREML=ON"
        BUILD_FLAG="--use_coreml"
        ;;
    all|auto)
        echo "[2/5] Configuring for all available providers..."
        CMAKE_ARGS="$CMAKE_ARGS \
            -DONNXRUNTIME_USE_CUDA=ON \
            -DONNXRUNTIME_USE_COREML=ON"
        BUILD_FLAG="--use_cuda --use_coreml"
        ;;
    *)
        echo "Unknown provider: $PROVIDER"
        echo "Usage: $0 [cuda|coreml|all]"
        exit 1
        ;;
esac

echo "[3/5] Building..."
./build.sh --config Release \
    --build_shared_lib \
    --skip_tests \
    --parallel \
    $BUILD_FLAG

echo "[4/5] Installing to $INSTALL_DIR..."
mkdir -p "$INSTALL_DIR/lib"
cp -L libRelease/*.so* "$INSTALL_DIR/lib/" 2>/dev/null || true
cp -L libRelease/*.dylib* "$INSTALL_DIR/lib/" 2>/dev/null || true
cp -r include/* "$INSTALL_DIR/include/" 2>/dev/null || true

echo "[5/5] Done!"
echo ""
echo "GPU-enabled ONNX Runtime libraries installed to:"
echo "  $INSTALL_DIR/lib/"
echo ""
echo "To use GPU acceleration, run linetime with:"
echo "  linetime song.wav lyrics.txt --gpu"
echo "  linetime song.wav lyrics.txt --provider cuda"
echo "  linetime song.wav lyrics.txt --provider coreml"
