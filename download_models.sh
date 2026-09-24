#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
MODELS_DIR="${SCRIPT_DIR}/models"

mkdir -p "$MODELS_DIR"

echo "=== linetime model downloader ==="
echo "Target directory: $MODELS_DIR"
echo ""

# --- MMS multilingual CTC model ---
echo "[1/3] Downloading MMS multilingual CTC model..."
echo "  Source: huggingface.co/xenova/mms_multilingual"
echo ""

MMS_BASE="https://huggingface.co/xenova/mms_multilingual/resolve/main"

if [ -f "$MODELS_DIR/mms_multilingual.onnx" ] && [ -f "$MODELS_DIR/mms_multilingual_tokenizer.json" ]; then
    echo "  [skip] mms_multilingual files already exist"
else
    wget -q --show-progress -P "$MODELS_DIR/" "${MMS_BASE}/quantize.onnx" -O "$MODELS_DIR/mms_multilingual.onnx" || true
    wget -q --show-progress -P "$MODELS_DIR/" "${MMS_BASE}/tokenizer.json" -O "$MODELS_DIR/mms_multilingual_tokenizer.json" || true
fi

echo ""

# --- Whisper large-v3 (Method C STT) ---
echo "[2/3] Downloading whisper.cpp large-v3 model (~3.1GB)..."
echo ""

WHISPER_BASE="https://huggingface.co/ggerganov/whisper.cpp/resolve/main"

if [ -f "$MODELS_DIR/ggml-large-v3.bin" ]; then
    echo "  [skip] ggml-large-v3.bin already exists"
else
    wget -q --show-progress -P "$MODELS_DIR/" "${WHISPER_BASE}/ggml-large-v3.bin"
fi

echo ""

# --- Verify ---
echo "[3/3] Verifying downloads..."
echo ""
echo "Files in $MODELS_DIR:"
ls -lh "$MODELS_DIR/"
echo ""

MISSING=0
for f in mms_multilingual.onnx mms_multilingual_tokenizer.json ggml-large-v3.bin; do
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
echo "  ./linetime <audio> <lyrics> --method a --model-a models/mms_multilingual.onnx --tokenizer models/mms_multilingual_tokenizer.json"
echo "  ./linetime <audio> <lyrics> --method c --model-c models/ggml-large-v3.bin --language <code>"