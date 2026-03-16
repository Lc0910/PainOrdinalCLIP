#!/usr/bin/env bash
# =============================================================================
# BioVid AU 提取完整流程：OpenFace 安装 → 视频帧批量处理 → NPZ 生成
#
# 适用环境：Ubuntu 服务器（已测试 18.04/20.04/22.04）
#
# 用法：
#   # 第一步：安装 OpenFace（只需执行一次）
#   bash scripts/data/run_openface_extract_au.sh install
#
#   # 第二步：从 BioVid 图像帧提取 AU → 生成 NPZ
#   bash scripts/data/run_openface_extract_au.sh extract \
#       --images_root data/biovid \
#       --data_file data/biovid/train_skip2.txt \
#       --output_dir data/biovid/openface_csv \
#       --npz_all data/biovid/au_features_openface_all17.npz \
#       --npz_pain data/biovid/au_features_openface_pain8.npz
#
#   # 可选：只跑 OpenFace（不生成 NPZ），用于调试
#   bash scripts/data/run_openface_extract_au.sh openface_only \
#       --images_root data/biovid \
#       --data_file data/biovid/train_skip2.txt \
#       --output_dir data/biovid/openface_csv
# =============================================================================

set -euo pipefail

OPENFACE_DIR="${OPENFACE_DIR:-$HOME/OpenFace}"
OPENFACE_BIN="${OPENFACE_DIR}/build/bin/FaceLandmarkImg"

# ─── Colors ──────────────────────────────────────────────────────────
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m'

log_info()  { echo -e "${GREEN}[INFO]${NC} $*"; }
log_warn()  { echo -e "${YELLOW}[WARN]${NC} $*"; }
log_error() { echo -e "${RED}[ERROR]${NC} $*"; }

# ─── Install OpenFace ────────────────────────────────────────────────
install_openface() {
    log_info "Installing OpenFace to ${OPENFACE_DIR}..."

    # System dependencies
    log_info "Installing system dependencies..."
    apt-get update
    apt-get install -y \
        build-essential cmake git \
        libopenblas-dev liblapack-dev \
        libopencv-dev \
        libboost-all-dev \
        wget unzip

    # Clone OpenFace
    if [ -d "${OPENFACE_DIR}" ]; then
        log_warn "OpenFace directory already exists: ${OPENFACE_DIR}"
        log_warn "Skipping clone. Delete it first if you want a fresh install."
    else
        log_info "Cloning OpenFace..."
        git clone https://github.com/TadasBaltrusaitis/OpenFace.git "${OPENFACE_DIR}"
    fi

    cd "${OPENFACE_DIR}"

    # Download models
    log_info "Downloading pre-trained models..."
    if [ -f "download_models.sh" ]; then
        bash download_models.sh
    elif [ -f "model/patch_experts" ] && [ -d "model/patch_experts" ]; then
        log_info "Models appear to already exist, skipping download."
    else
        log_warn "download_models.sh not found. You may need to download models manually."
        log_warn "See: https://github.com/TadasBaltrusaitis/OpenFace/wiki/Model-download"
    fi

    # Build
    log_info "Building OpenFace..."
    mkdir -p build && cd build
    cmake -D CMAKE_BUILD_TYPE=RELEASE ..
    make -j$(nproc)

    # Verify
    if [ -f "${OPENFACE_BIN}" ]; then
        log_info "OpenFace installed successfully!"
        log_info "Binary: ${OPENFACE_BIN}"
        "${OPENFACE_BIN}" --help 2>&1 | head -3 || true
    else
        log_error "Build failed — ${OPENFACE_BIN} not found"
        exit 1
    fi
}

# ─── Run OpenFace on image frames ───────────────────────────────────
# Delegates to run_openface_batch.py which correctly handles frame number
# mapping in the merged per-video CSVs.  The old shell-based merge used
# `tail -n +2` and kept OpenFace's default frame=1, causing extract_au_features.py
# to mis-match frames when building the .npz lookup table.
run_openface() {
    local images_root="$1"
    local data_file="$2"
    local output_dir="$3"

    if [ ! -f "${OPENFACE_BIN}" ]; then
        log_error "OpenFace not found at ${OPENFACE_BIN}"
        log_error "Run: bash $0 install"
        exit 1
    fi

    local script_dir
    script_dir="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

    python "${script_dir}/run_openface_batch.py" \
        --images_root "${images_root}" \
        --data_file "${data_file}" \
        --output_dir "${output_dir}" \
        --openface_dir "${OPENFACE_DIR}"
}

# ─── Generate NPZ from OpenFace CSVs ────────────────────────────────
generate_npz() {
    local data_file="$1"
    local output_dir="$2"
    local npz_all="$3"
    local npz_pain="$4"

    log_info "Generating NPZ files from OpenFace CSVs..."

    # 17 AUs (all)
    python scripts/data/extract_au_features.py \
        --openface_dir "${output_dir}" \
        --data_file "${data_file}" \
        --output "${npz_all}" \
        --au_subset all

    # 8 AUs (pain-relevant)
    python scripts/data/extract_au_features.py \
        --openface_dir "${output_dir}" \
        --data_file "${data_file}" \
        --output "${npz_pain}" \
        --au_subset pain

    log_info "NPZ generation complete!"
    log_info "  All 17 AUs: ${npz_all}"
    log_info "  Pain 8 AUs: ${npz_pain}"
}

# ─── Quality comparison: py-feat vs OpenFace ─────────────────────────
compare_au_sources() {
    local npz_pyfeat="$1"
    local npz_openface="$2"
    local data_file="$3"
    local au_dim="$4"

    log_info "Comparing py-feat vs OpenFace AU features..."

    python3 - "${npz_pyfeat}" "${npz_openface}" "${data_file}" "${au_dim}" <<'PYEOF'
import sys
import numpy as np

npz_pyfeat_path = sys.argv[1]
npz_openface_path = sys.argv[2]
data_file = sys.argv[3]
au_dim = int(sys.argv[4])

pf = np.load(npz_pyfeat_path, allow_pickle=False)
of = np.load(npz_openface_path, allow_pickle=False)

pf_keys = set(pf.files)
of_keys = set(of.files)

# Read data list
with open(data_file) as f:
    dl_keys = set(line.strip().split()[0] for line in f if line.strip())

pf_coverage = len(pf_keys & dl_keys) / len(dl_keys) * 100
of_coverage = len(of_keys & dl_keys) / len(dl_keys) * 100
both_keys = pf_keys & of_keys & dl_keys

print(f"\nData list frames:    {len(dl_keys)}")
print(f"py-feat coverage:    {len(pf_keys & dl_keys)} ({pf_coverage:.1f}%)")
print(f"OpenFace coverage:   {len(of_keys & dl_keys)} ({of_coverage:.1f}%)")
print(f"Both have:           {len(both_keys)}")

if len(both_keys) == 0:
    print("No overlapping keys — cannot compare.")
    sys.exit(0)

# Compare AU values on shared frames
pf_vecs = np.stack([pf[k][:au_dim] for k in sorted(both_keys)])
of_vecs = np.stack([of[k][:au_dim] for k in sorted(both_keys)])

diff = np.abs(pf_vecs - of_vecs)
print(f"\nPer-dimension comparison (N={len(both_keys)} shared frames):")
print(f"{'Dim':>4s}  {'pf_mean':>8s}  {'of_mean':>8s}  {'pf_std':>7s}  {'of_std':>7s}  {'MAE':>7s}  {'corr':>7s}")
for i in range(min(au_dim, pf_vecs.shape[1], of_vecs.shape[1])):
    pf_m = pf_vecs[:, i].mean()
    of_m = of_vecs[:, i].mean()
    pf_s = pf_vecs[:, i].std()
    of_s = of_vecs[:, i].std()
    mae = diff[:, i].mean()
    if pf_s > 1e-6 and of_s > 1e-6:
        corr = np.corrcoef(pf_vecs[:, i], of_vecs[:, i])[0, 1]
    else:
        corr = float('nan')
    print(f"  {i:2d}  {pf_m:8.4f}  {of_m:8.4f}  {pf_s:7.4f}  {of_s:7.4f}  {mae:7.4f}  {corr:7.4f}")

overall_corr = []
for i in range(min(au_dim, pf_vecs.shape[1], of_vecs.shape[1])):
    pf_s = pf_vecs[:, i].std()
    of_s = of_vecs[:, i].std()
    if pf_s > 1e-6 and of_s > 1e-6:
        overall_corr.append(np.corrcoef(pf_vecs[:, i], of_vecs[:, i])[0, 1])
if overall_corr:
    print(f"\nMean correlation across dimensions: {np.mean(overall_corr):.4f}")
else:
    print("\nCannot compute correlation (constant features)")
PYEOF
}

# ─── Main ────────────────────────────────────────────────────────────
main() {
    local cmd="${1:-help}"
    shift || true

    case "${cmd}" in
        install)
            install_openface
            ;;

        extract)
            # Parse named arguments
            local images_root="" data_file="" output_dir=""
            local npz_all="data/biovid/au_features_openface_all17.npz"
            local npz_pain="data/biovid/au_features_openface_pain8.npz"

            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --images_root) images_root="$2"; shift 2;;
                    --data_file)   data_file="$2";   shift 2;;
                    --output_dir)  output_dir="$2";  shift 2;;
                    --npz_all)     npz_all="$2";     shift 2;;
                    --npz_pain)    npz_pain="$2";    shift 2;;
                    *) log_error "Unknown argument: $1"; exit 1;;
                esac
            done

            if [ -z "${images_root}" ] || [ -z "${data_file}" ] || [ -z "${output_dir}" ]; then
                log_error "Usage: $0 extract --images_root DIR --data_file FILE --output_dir DIR [--npz_all FILE] [--npz_pain FILE]"
                exit 1
            fi

            run_openface "${images_root}" "${data_file}" "${output_dir}"
            generate_npz "${data_file}" "${output_dir}" "${npz_all}" "${npz_pain}"

            log_info ""
            log_info "Done! Next steps:"
            log_info "  # Run AU-only baseline with OpenFace features"
            log_info "  python scripts/diagnosis/au_only_baseline.py \\"
            log_info "      --au_npz ${npz_pain} --au_dim 8 \\"
            log_info "      --output_dir results/au-only-baseline-openface-pain8"
            log_info ""
            log_info "  # Compare py-feat vs OpenFace"
            log_info "  bash $0 compare \\"
            log_info "      --pyfeat data/biovid/au_features_pain8.npz \\"
            log_info "      --openface ${npz_pain} \\"
            log_info "      --data_file ${data_file} --au_dim 8"
            ;;

        openface_only)
            local images_root="" data_file="" output_dir=""
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --images_root) images_root="$2"; shift 2;;
                    --data_file)   data_file="$2";   shift 2;;
                    --output_dir)  output_dir="$2";  shift 2;;
                    *) log_error "Unknown argument: $1"; exit 1;;
                esac
            done

            if [ -z "${images_root}" ] || [ -z "${data_file}" ] || [ -z "${output_dir}" ]; then
                log_error "Usage: $0 openface_only --images_root DIR --data_file FILE --output_dir DIR"
                exit 1
            fi

            run_openface "${images_root}" "${data_file}" "${output_dir}"
            ;;

        compare)
            local pyfeat="" openface="" data_file="" au_dim=8
            while [[ $# -gt 0 ]]; do
                case "$1" in
                    --pyfeat)    pyfeat="$2";    shift 2;;
                    --openface)  openface="$2";  shift 2;;
                    --data_file) data_file="$2"; shift 2;;
                    --au_dim)    au_dim="$2";    shift 2;;
                    *) log_error "Unknown argument: $1"; exit 1;;
                esac
            done

            if [ -z "${pyfeat}" ] || [ -z "${openface}" ] || [ -z "${data_file}" ]; then
                log_error "Usage: $0 compare --pyfeat NPZ --openface NPZ --data_file FILE [--au_dim N]"
                exit 1
            fi

            compare_au_sources "${pyfeat}" "${openface}" "${data_file}" "${au_dim}"
            ;;

        help|*)
            echo "Usage: $0 <command> [options]"
            echo ""
            echo "Commands:"
            echo "  install         Install OpenFace on Ubuntu (requires sudo)"
            echo "  extract         Run OpenFace + generate NPZ files"
            echo "  openface_only   Run OpenFace only (no NPZ generation)"
            echo "  compare         Compare py-feat vs OpenFace AU features"
            echo ""
            echo "Environment:"
            echo "  OPENFACE_DIR    OpenFace install path (default: ~/OpenFace)"
            echo ""
            echo "Examples:"
            echo "  # Install OpenFace"
            echo "  bash $0 install"
            echo ""
            echo "  # Full pipeline: OpenFace → CSV → NPZ"
            echo "  bash $0 extract \\"
            echo "      --images_root data/biovid \\"
            echo "      --data_file data/biovid/train_skip2.txt \\"
            echo "      --output_dir data/biovid/openface_csv \\"
            echo "      --npz_all data/biovid/au_features_openface_all17.npz \\"
            echo "      --npz_pain data/biovid/au_features_openface_pain8.npz"
            echo ""
            echo "  # Compare py-feat vs OpenFace"
            echo "  bash $0 compare \\"
            echo "      --pyfeat data/biovid/au_features_pain8.npz \\"
            echo "      --openface data/biovid/au_features_openface_pain8.npz \\"
            echo "      --data_file data/biovid/train_skip2.txt --au_dim 8"
            ;;
    esac
}

main "$@"
