#!/bin/bash
#SBATCH --job-name=deeplabv3      # Job name
#SBATCH --output=logs/job_%j.log    # Standard output/error log (%j = JobID)
#SBATCH --partition=main            # Partition/Queue name (e.g., short, gpu) [1, 3]
#SBATCH --gres=gpu:b200_full:1   # Request 1 slice of the 23GB instance
#SBATCH --cpus-per-task=64
#SBATCH --mem=128G                  # Job memory request (e.g., 16G, 100M) [14]
#SBATCH --time=24:00:00             # Time limit (HH:MM:SS) [12]

source ~/miniconda3/etc/profile.d/conda.sh
conda activate persona

# export CUDA_VISIBLE_DEVICES=$(nvidia-smi -L | grep MIG | awk '{print $NF}' | tr -d ')' | head -n 1)
# echo "Using MIG Device: $CUDA_VISIBLE_DEVICES"

export CUDA_HOME=$CONDA_PREFIX
export PATH=$CUDA_HOME/bin:$PATH

echo $CUDA_HOME

export TORCH_CUDA_ARCH_LIST="10.0"

mkdir -p ~/tmp
export TMPDIR=~/tmp

export MAX_JOBS=64

pip install flash-attn --no-build-isolation
pip install bitsandbytes accelerate autoawq

#python dataset_construction/run_pipelines.py

