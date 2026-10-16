#!/usr/bin/env bash
# Submit from the SocialJev checkout on Clariden; --time is an integer in minutes.
set -euo pipefail
if [[ ${1:-} != --worker ]]; then
    cd "$(dirname "${BASH_SOURCE[0]}")/.."
    minutes=5
    benchmark_args=()
    while (( $# )); do
        case $1 in
            --time) minutes=${2:?--time requires minutes}; shift 2 ;;
            *) benchmark_args+=("$1"); shift ;;
        esac
    done
    [[ $minutes =~ ^[1-9][0-9]*$ ]]
    mkdir -p artifacts
    sbatch --account=abs01 --partition=debug --nodes=1 --ntasks=1 \
        --gpus-per-task=1 --time="$minutes" --job-name=socialjev-wikipedia \
        --chdir="$PWD" --output="$PWD/artifacts/wikipedia-%j.log" \
        scripts/wikipedia.sh --worker "${benchmark_args[@]}"
    exit
fi
shift

export XDG_CACHE_HOME="${SCRATCH:?}/.cache"
export HF_HOME="$XDG_CACHE_HOME/huggingface" UV_CACHE_DIR="$XDG_CACHE_HOME/uv"
export HF_HUB_CACHE="$HF_HOME/hub" HF_DATASETS_CACHE="$HF_HOME/datasets"
export TORCHINDUCTOR_CACHE_DIR="$XDG_CACHE_HOME/torchinductor"
export TRITON_CACHE_DIR="$XDG_CACHE_HOME/triton" CUDA_CACHE_PATH="$XDG_CACHE_HOME/nvidia/ComputeCache"
export TMPDIR="$SCRATCH/social-jev/tmp"
# Enroot runtime state is per allocation; downloads and temporary files are shared.
export XDG_RUNTIME_DIR="$SCRATCH/social-jev/runtime/$SLURM_JOB_ID"
export ENROOT_RUNTIME_PATH="$XDG_RUNTIME_DIR/enroot"
export ENROOT_CACHE_PATH="$XDG_CACHE_HOME/enroot"
export ENROOT_DATA_PATH="$SCRATCH/social-jev/enroot/data"
export ENROOT_TEMP_PATH="$TMPDIR"
mkdir -p "$TMPDIR" "$XDG_RUNTIME_DIR" "$ENROOT_CACHE_PATH" "$ENROOT_DATA_PATH"
chmod 700 "$XDG_RUNTIME_DIR"

# Variables in bash -c expand inside the container.
# shellcheck disable=SC2016
srun --ntasks=1 --gpus-per-task=1 --cpu-bind=none --network=disable_rdzv_get \
    --environment="$PWD/containers/clariden.toml" --container-workdir="$PWD" \
    bash -c '
        set -euo pipefail
        export CUDA_VISIBLE_DEVICES=0
        export UV_PROJECT_ENVIRONMENT="$SCRATCH/social-jev/venv-wikipedia"
        /usr/local/bin/uv sync --locked --group benchmark --extra vllm
        exec /usr/local/bin/uv run --no-sync python scripts/wikipedia.py "$@"
    ' wikipedia "$@"
