#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --gres=gpu:nvidia_b300_sxm6_ac:1
#SBATCH --time=100:00:00
#SBATCH --job-name=touche-gemma4-v2
#SBATCH --output=touche_gemma4_v2-%j.out
#SBATCH --error=touche_gemma4_v2-%j.err

set -euo pipefail

# Slurm often runs a *copy* of this script under /var/tmp, so BASH_SOURCE is not under the repo.
# SLURM_SUBMIT_DIR is the directory you were in when you ran `sbatch` — use that as repo root.
if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

# Conda env: steering-b300
if [[ -f "${CONDA_EXE:-}" ]] && [[ -d "$(dirname "$CONDA_EXE")/../etc/profile.d" ]]; then
  source "$(dirname "$CONDA_EXE")/../etc/profile.d/conda.sh"
elif [[ -f "$HOME/miniconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/miniconda3/etc/profile.d/conda.sh"
elif [[ -f "$HOME/anaconda3/etc/profile.d/conda.sh" ]]; then
  source "$HOME/anaconda3/etc/profile.d/conda.sh"
elif [[ -f "/opt/conda/etc/profile.d/conda.sh" ]]; then
  source "/opt/conda/etc/profile.d/conda.sh"
else
  echo "Could not find conda.sh; set CONDA_EXE or edit this script." >&2
  exit 1
fi

conda activate steering-b300

python dataset_construction/Touche23-ValueEval/run_pipelines.py \
  --input dataset_construction/Touche23-ValueEval/data/touche_positive_only_sampled_200.csv \
  --output dataset_construction/Touche23-ValueEval/data/touche_gemma4-v2.csv
