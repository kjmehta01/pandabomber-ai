#!/usr/bin/env bash
# install-cuda-wsl.sh
#
# Installs CUDA 11.8 + cuDNN 8.6 inside WSL2 for tfjs-node-gpu 4.x.
# These specific versions are pinned by tfjs-node-gpu — don't substitute.
#
# Assumes the NVIDIA driver is already installed on the Windows host (verify
# with `nvidia-smi` before running). Driver is NOT installed in WSL2 — only
# the toolkit and cuDNN libraries.
#
# Idempotent: each step skips if already done. Safe to re-run.
#
# cuDNN is login-gated; if no tarball is found the script prints instructions
# and exits cleanly so you can re-run after downloading.

set -euo pipefail

CUDA_VERSION=11.8
CUDA_INSTALL_DIR=/usr/local/cuda-${CUDA_VERSION}
CUDA_RUN_URL=https://developer.download.nvidia.com/compute/cuda/11.8.0/local_installers/cuda_11.8.0_520.61.05_linux.run
CUDA_RUN_FILE=/tmp/cuda_11.8.0_520.61.05_linux.run

log() { echo "[install-cuda] $*"; }
die() { echo "[install-cuda] ERROR: $*" >&2; exit 1; }

# 1. Preflight: driver visible from WSL2.
command -v nvidia-smi >/dev/null 2>&1 || die "nvidia-smi not found. Install the NVIDIA Windows driver on the host first."
nvidia-smi >/dev/null 2>&1            || die "nvidia-smi failed. Driver/passthrough is broken."
log "GPU detected: $(nvidia-smi --query-gpu=name --format=csv,noheader,nounits | head -1)"

# 2. CUDA 11.8 toolkit.
if [[ -f "${CUDA_INSTALL_DIR}/lib64/libcudart.so.11.0" ]]; then
    log "CUDA ${CUDA_VERSION} already installed at ${CUDA_INSTALL_DIR}"
else
    if [[ ! -f "${CUDA_RUN_FILE}" ]]; then
        log "downloading CUDA ${CUDA_VERSION} installer (~3 GB)..."
        wget -O "${CUDA_RUN_FILE}" "${CUDA_RUN_URL}"
    else
        log "CUDA installer already at ${CUDA_RUN_FILE}; reusing"
    fi
    log "installing CUDA toolkit (host driver kept; --toolkit only)..."
    sudo sh "${CUDA_RUN_FILE}" --silent --toolkit --override
    [[ -f "${CUDA_INSTALL_DIR}/lib64/libcudart.so.11.0" ]] || die "CUDA install ran but libcudart.so.11.0 missing. Check /var/log/cuda-installer.log"
    log "CUDA ${CUDA_VERSION} installed"
fi

# 3. PATH / LD_LIBRARY_PATH in ~/.bashrc (idempotent — marker line prevents duplicates).
BASHRC=~/.bashrc
MARKER="# cuda-11.8 paths (added by install-cuda-wsl.sh)"
if ! grep -qF "${MARKER}" "${BASHRC}"; then
    log "adding CUDA paths to ${BASHRC}"
    cat >> "${BASHRC}" <<EOF

${MARKER}
export PATH=${CUDA_INSTALL_DIR}/bin:\$PATH
export LD_LIBRARY_PATH=${CUDA_INSTALL_DIR}/lib64:\${LD_LIBRARY_PATH:-}
EOF
else
    log "CUDA paths already in ${BASHRC}"
fi
# Export for this shell too so the verification step at the bottom works.
export PATH=${CUDA_INSTALL_DIR}/bin:$PATH
export LD_LIBRARY_PATH=${CUDA_INSTALL_DIR}/lib64:${LD_LIBRARY_PATH:-}

# 4. cuDNN 8.6 — login-gated, can't be wget'd directly.
if [[ -f "${CUDA_INSTALL_DIR}/lib64/libcudnn.so.8" ]]; then
    log "cuDNN 8 already installed in ${CUDA_INSTALL_DIR}/lib64"
else
    CUDNN_TAR=${CUDNN_TAR:-}
    if [[ -z "${CUDNN_TAR}" ]]; then
        # Auto-detect tarball in cwd or ~/Downloads.
        for pattern in \
            "./cudnn-linux-x86_64-8.6.0.*_cuda11-archive.tar.xz" \
            "${HOME}/Downloads/cudnn-linux-x86_64-8.6.0.*_cuda11-archive.tar.xz"; do
            match=$(compgen -G "${pattern}" 2>/dev/null | head -1 || true)
            if [[ -n "${match}" ]]; then
                CUDNN_TAR="${match}"
                break
            fi
        done
    fi
    if [[ -z "${CUDNN_TAR}" || ! -f "${CUDNN_TAR}" ]]; then
        cat <<EOF

cuDNN 8.6 tarball not found. NVIDIA gates this behind a developer account:
  1. https://developer.nvidia.com/rdp/cudnn-archive
  2. Sign in (free, approval immediate)
  3. Find: "cuDNN v8.6.0 (October 3rd, 2022), for CUDA 11.x"
  4. Download: "Local Installer for Linux x86_64 (Tar)"
  5. Move the .tar.xz into this directory or ~/Downloads, then re-run this script.
     Or pass an explicit path:  CUDNN_TAR=/path/to/cudnn-...tar.xz $0

CUDA toolkit is installed and PATH is set; only cuDNN is missing.
EOF
        exit 0
    fi
    log "extracting ${CUDNN_TAR}..."
    EXTRACT_DIR=$(mktemp -d)
    tar -xJf "${CUDNN_TAR}" -C "${EXTRACT_DIR}"
    CUDNN_SRC=$(find "${EXTRACT_DIR}" -maxdepth 2 -type d -name 'cudnn-linux-*' | head -1)
    [[ -d "${CUDNN_SRC}" ]] || die "couldn't find extracted cuDNN dir inside ${EXTRACT_DIR}"
    sudo cp -P "${CUDNN_SRC}/include/"cudnn*.h "${CUDA_INSTALL_DIR}/include/"
    sudo cp -P "${CUDNN_SRC}/lib/"libcudnn*    "${CUDA_INSTALL_DIR}/lib64/"
    sudo chmod a+r "${CUDA_INSTALL_DIR}/include/"cudnn*.h "${CUDA_INSTALL_DIR}/lib64/"libcudnn*
    rm -rf "${EXTRACT_DIR}"
    log "cuDNN 8.6 installed"
fi

# 5. Verify.
log "verifying..."
[[ -f "${CUDA_INSTALL_DIR}/lib64/libcudart.so.11.0" ]] || die "libcudart.so.11.0 missing"
[[ -f "${CUDA_INSTALL_DIR}/lib64/libcudnn.so.8" ]]      || die "libcudnn.so.8 missing"
nvcc --version | grep -q "release 11.8"                 || die "nvcc not reporting 11.8"
log "all libs present, nvcc reports 11.8"
log ""
log "Open a new shell (or run: source ~/.bashrc) so PATH/LD_LIBRARY_PATH update, then:"
log "    cd ~/bomberman/pandabomber-ai"
log "    npm run train:gpu -- --episodes=2000 --save=checkpoints/latest.json --resume=off"
log ""
log "Confirm GPU is actually active by looking for this line in the startup logs:"
log "    Created device /job:localhost/replica:0/task:0/device:GPU:0 with N MB memory"
