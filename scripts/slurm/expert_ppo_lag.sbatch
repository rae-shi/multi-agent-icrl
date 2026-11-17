#!/bin/bash
#SBATCH --account=def-s2ganapa
#SBATCH --job-name=expert_ppo_lag
#SBATCH --time=10:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --array=0-2
#SBATCH --output=%x_%A_%a.out
# Real PPO-Lag expert for MultiGoal level 1, two-row env. Budget 0, as in every upstream train_ppo_lag config.
# One array task per seed; seeds are taken from the list below (upstream seeds, first three).
# Run length comes from the config: n_iters x forward_timesteps rows (50 x 200000, rounded up to whole rollouts).
# Submit from the repo root:  sbatch [--array=0] scripts/slurm/expert_ppo_lag.sbatch [config.yaml]
# Config: first argument, else the CFG environment variable, else the default config below.
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/projects/def-s2ganapa/raeshi/venv_safety/bin/activate
export PYTHONPATH="$PWD:$PWD/algos/icrl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # 2.6x faster in the profile

SEEDS=(123 321 456 654 666)
SEED=${SEEDS[$SLURM_ARRAY_TASK_ID]}
CFG=${1:-${CFG:-configs/multigoal/train_ppo_lag_SafetyAntMultiGoal1Pair-v0.yaml}}   # 1st argument, else $CFG, else default
[ -f "$CFG" ] || { echo "config not found: $CFG"; exit 1; }
echo "config: $CFG"

echo "commit: $(git rev-parse HEAD)"
git status --short | head -5
echo "seed: $SEED  array task: $SLURM_ARRAY_TASK_ID  host: $(hostname)"
echo "python: $(which python)"
grep -n "^task\|budget\|n_iters\|forward_timesteps\|n_steps\|n_epochs\|learning_rate\|penalty" "$CFG"
cp "$CFG" "$SLURM_SUBMIT_DIR/expert_ppo_lag_config_${SLURM_ARRAY_JOB_ID}_${SEED}.yaml"

cd algos/icrl/interface
python train_policy.py ../../../"$CFG" -n 8 -s "$SEED"
