#!/usr/bin/env bash
#SBATCH --partition=main
#SBATCH --gres=gpu:nvidia_b300_sxm6_ac:1
#SBATCH --time=100:00:00
#SBATCH --job-name=touche-gemma4-v2-val
#SBATCH --output=touche_gemma4_v2_validate-%j.out
#SBATCH --error=touche_gemma4_v2_validate-%j.err

set -euo pipefail

if [[ -n "${SLURM_SUBMIT_DIR:-}" ]]; then
  ROOT="$SLURM_SUBMIT_DIR"
else
  ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
fi
cd "$ROOT"

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

python dataset_construction/Touche23-ValueEval/validate.py \
  --input dataset_construction/Touche23-ValueEval/data/touche_gemma4-v2_remaining.csv \
  --output dataset_construction/Touche23-ValueEval/data/touche_gemma4-v2_remaining-validated.csv \
  --sample 0 \
  --method batch \
  --batch-size 64
