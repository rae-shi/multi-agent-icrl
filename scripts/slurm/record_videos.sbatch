#!/bin/bash
#SBATCH --account=def-s2ganapa
#SBATCH --job-name=record_videos
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --output=record_videos_%j.out

# Usage: sbatch scripts/slurm/record_videos.sbatch [RUN_DIR] [OUT_DIR]
# Submit from the repo root.

set -eo pipefail
module load StdEnv/2023 glfw/3.3.8
export PYGLFW_LIBRARY=$(ls $EBROOTGLFW/lib*/libglfw.so* | head -1)
source ~/projects/def-s2ganapa/raeshi/venv_safety/bin/activate      # your actual venv path
cd "$SLURM_SUBMIT_DIR"                 # repo root = where you ran sbatch

export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
export PYTHONPATH="$PWD:$PWD/algos/icrl:$PYTHONPATH"

RUN=${1:-$(ls -d runs/PPO-Lag-MG1-Pair-budget1p0/*/ | head -1)}
OUT=${2:-videos/budget1p0}
[ -d "$RUN" ] || { echo "run dir not found: $RUN"; exit 1; }
echo "commit: $(git rev-parse --short HEAD) | run: $RUN | out: $OUT"

python scripts/env_checks/record_policy_video.py --run-dir "$RUN" \
  --iterations 10 20 30 40 --episodes 2 --out "$OUT"
python scripts/env_checks/record_policy_video.py --run-dir "$RUN" \
  --iterations 10 20 30 40 --episodes 2 --stochastic --out "$OUT"