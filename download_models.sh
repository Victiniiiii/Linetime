#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"

mkdir -p "$MODELS_DIR"

echo "=== linetime model downloader ==="
echo "Target directory: $MODELS_DIR"
echo ""

# --- MMS_FA CTC model ---
echo "[1/3] Downloading MMS_FA CTC model..."
echo "  Source: huggingface.co/xycld/lyric-align-mms-fa"
echo "  NOTE: This model is primarily trained on Chinese."
echo "        For non-Chinese languages, use --method b (whisper only)."
echo ""

MMS_FA_BASE="https://huggingface.co/xycld/lyric-align-mms-fa/resolve/main"

if [ -f "$MODELS_DIR/mms_fa.onnx" ] && [ -f "$MODELS_DIR/mms_fa.onnx.data" ] && [ -f "$MODELS_DIR/tokenizer.json" ]; then
    echo "  [skip] MMS_FA files already exist"
else
    wget -q --show-progress -P "$MODELS_DIR/" "${MMS_FA_BASE}/tokenizer.json" || true
    wget -q --show-progress -P "$MODELS_DIR/" "${MMS_FA_BASE}/mms_fa.onnx" || true
    wget -q --show-progress -P "$MODELS_DIR/" "${MMS_FA_BASE}/mms_fa.onnx.data" || true
fi

echo ""

# --- Whisper model ---
echo "[2/3] Downloading whisper.cpp model..."
echo "  Using ggml-small.bin (466MB) — better multilingual accuracy"
echo "  For faster inference with acceptable quality, use ggml-base.bin instead."
echo ""

WHISPER_BASE="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

if [ -f "$MODELS_DIR/ggml-small.bin" ]; then
    echo "  [skip] ggml-small.bin already exists"
else
    wget -q --show-progress -P "$MODELS_DIR/" "${WHISPER_BASE}/ggml-small.bin"
fi

# Also download base as fallback
if [ -f "$MODELS_DIR/ggml-base.bin" ]; then
    echo "  [skip] ggml-base.bin already exists"
else
    echo ""
    echo "  Also downloading ggml-base.bin (142MB) as fallback..."
    wget -q --show-progress -P "$MODELS_DIR/" "${WHISPER_BASE}/ggml-base.bin"
fi

echo ""

# --- Verify ---
echo "[3/3] Verifying downloads..."
echo ""
echo "Files in $MODELS_DIR:"
ls -lh "$MODELS_DIR/"
echo ""

MISSING=0
for f in mms_fa.onnx mms_fa.onnx.data tokenizer.json ggml-small.bin ggml-base.bin; do
    if [ ! -f "$MODELS_DIR/$f" ]; then
        echo "  MISSING: $f"
        MISSING=1
    fi
done

if [ "$MISSING" -eq 0 ]; then
    echo "  All models present."
else
    echo "  Some models failed to download. Check your network connection."
fi

echo ""
echo "Done. Run with:"
echo "  ./build/linetime <audio> <lyrics> --method b --language <code>"
echo "  (Use --method b for non-Chinese languages)"
