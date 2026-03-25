#!/bin/bash
# Download FER pretrained weights for PainOrdinalCLIP experiments.
#
# Usage:
#   bash scripts/download_fer_weights.sh          # download all
#   bash scripts/download_fer_weights.sh dan       # DAN only
#   bash scripts/download_fer_weights.sh hsemotion # HSEmotion only
#
# Prerequisites:
#   pip install gdown    # for Google Drive downloads
#   pip install hsemotion # for HSEmotion auto-download

set -e

CACHE_DIR=".cache/fer"
mkdir -p "$CACHE_DIR"

download_dan() {
    echo "=========================================="
    echo "Downloading DAN weights..."
    echo "=========================================="

    if [ -f "$CACHE_DIR/dan_affecnet7.pth" ]; then
        echo "  Already exists: $CACHE_DIR/dan_affecnet7.pth"
        return
    fi

    if ! command -v gdown &> /dev/null; then
        echo "ERROR: gdown not installed. Run: pip install gdown"
        echo ""
        echo "Manual download alternatives:"
        echo "  AffectNet-7: https://drive.google.com/file/d/1_Z-U7rT5NJ3Vc73aN2ZBmuvCkzUQG4jT"
        echo "  AffectNet-8: https://drive.google.com/file/d/1uHNADViICyJEjJljv747nfvrGu12kjtu"
        echo "  RAF-DB:      https://drive.google.com/file/d/1ASabP5wkLUIh4VQc8CEuZbZyLJEFaTMF"
        echo ""
        echo "  Baidu: https://pan.baidu.com/s/1NL-Yhuw5hF19uDWu-cF2-A (code: 0000)"
        echo ""
        echo "After download, place the file at: $CACHE_DIR/dan_affecnet7.pth"
        exit 1
    fi

    # AffectNet-7 checkpoint (~62-66% accuracy on AffectNet-7)
    echo "  Downloading AffectNet-7 checkpoint..."
    gdown 1_Z-U7rT5NJ3Vc73aN2ZBmuvCkzUQG4jT -O "$CACHE_DIR/dan_affecnet7.pth"

    echo "  Done: $CACHE_DIR/dan_affecnet7.pth"
    echo ""

    # Optional: RAF-DB checkpoint (~89.7% on RAF-DB)
    # Uncomment to download:
    # echo "  Downloading RAF-DB checkpoint..."
    # gdown 1ASabP5wkLUIh4VQc8CEuZbZyLJEFaTMF -O "$CACHE_DIR/dan_rafdb.pth"
}

download_hsemotion() {
    echo "=========================================="
    echo "Setting up HSEmotion weights..."
    echo "=========================================="

    if ! python -c "import hsemotion" 2>/dev/null; then
        echo "  Installing hsemotion..."
        pip install hsemotion
    fi

    # Trigger weight download by instantiating the model
    echo "  Triggering weight download for enet_b0_8_best_afew..."
    python -c "
from hsemotion.facial_emotions import HSEmotionRecognizer
fer = HSEmotionRecognizer(model_name='enet_b0_8_best_afew', device='cpu')
print('  HSEmotion B0 weights downloaded and cached.')
"
    echo "  Done. Weights cached in ~/.hsemotion/"
    echo ""
}

install_timm() {
    if ! python -c "import timm" 2>/dev/null; then
        echo "  Installing timm..."
        pip install "timm>=0.6.0"
    fi
}

# ----- Main -----

TARGET="${1:-all}"

case "$TARGET" in
    dan)
        download_dan
        ;;
    hsemotion)
        install_timm
        download_hsemotion
        ;;
    all)
        download_dan
        install_timm
        download_hsemotion
        ;;
    *)
        echo "Usage: $0 [dan|hsemotion|all]"
        exit 1
        ;;
esac

echo "=========================================="
echo "Weight download complete."
echo ""
echo "Available encoders:"
[ -f "$CACHE_DIR/dan_affecnet7.pth" ] && echo "  [OK] DAN:       $CACHE_DIR/dan_affecnet7.pth"
[ -f "$CACHE_DIR/dan_rafdb.pth" ]     && echo "  [OK] DAN-RAFDB: $CACHE_DIR/dan_rafdb.pth"
python -c "import hsemotion" 2>/dev/null && echo "  [OK] HSEmotion: auto-download via pip"
python -c "import timm" 2>/dev/null     && echo "  [OK] timm:      installed (for ViT-FER)"
echo "=========================================="
