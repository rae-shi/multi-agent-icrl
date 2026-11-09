#!/bin/bash
#SBATCH --account=def-s2ganapa
#SBATCH --job-name=pilot_unconstrained
#SBATCH --time=02:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --output=%x_%j.out
# Unconstrained pilot: PPO-Lag with budget 1.0 (a per-step fraction can never exceed it),
# so the dual penalty decays and the policy is essentially reward-only.
# Purpose: measure train/average_cost = fraction of (row, step) pairs with a violation, pooled over both agents.
# Submit from the repo root:  sbatch scripts/slurm/pilot_unconstrained.sbatch
set -eo pipefail
cd "$SLURM_SUBMIT_DIR"
source ~/projects/def-s2ganapa/raeshi/venv_safety/bin/activate
export PYTHONPATH="$PWD:$PWD/algos/icrl:${PYTHONPATH:-}"
export OMP_NUM_THREADS=1 MKL_NUM_THREADS=1   # 2.6x faster in the profile

echo "commit: $(git rev-parse HEAD)"
git status --short | head -5

SRC=configs/multigoal/train_ppo_lag_SafetyAntMultiGoal1Pair-v0.yaml
DST=configs/multigoal/pilot_unconstrained.yaml
python scripts/pipeline/make_pilot_config.py --src "$SRC" --dst "$DST" \
    --task PPO-Lag-MG1-Pair-pilot --budget 1.0 --n-iters 5
cp "$DST" "$SLURM_SUBMIT_DIR/pilot_unconstrained_config_${SLURM_JOB_ID}.yaml"

cd algos/icrl/interface
python train_policy.py ../../../"$DST" -n 8 -s 123
