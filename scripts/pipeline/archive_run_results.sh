#!/bin/bash
# Build one self-contained results folder for an expert run: docs/results/<TAG>/
#   config_used.yaml   the config the job actually read (copy made by the sbatch)
#   log_header.txt     first 20 lines of the job log (commit, config, seed, host, key settings)
#   train_log.out.gz   full log, only with --full-log
#   figures/  data/    the three figures and the CSVs behind them (from plot_expert_run.py)
#
# Usage (repo root, venv active):
#   scripts/pipeline/archive_run_results.sh TAG JOBID RUN_DIR BUDGET EVAL_GLOB [--full-log]
# Example:
#   scripts/pipeline/archive_run_results.sh budget5p4 4197959 "$RUN54" 0.054 "runs/expert_data/b5p4_it*_n30"
# Env: SEED (default 123), PLOT (default scripts/plots/plot_expert_run.py), RANDOM_REF (random-policy violation rate per agent, fractions; default "0.048 0.167" = Ant)
set -eo pipefail
[ $# -ge 5 ] || { sed -n 2,13p "$0"; exit 1; }
TAG=$1; JOB=$2; RUN=$3; BUDGET=$4; GLOB=$5; FULL=${6:-}
SEED=${SEED:-123}
PLOT=${PLOT:-scripts/plots/plot_expert_run.py}
LOG=expert_ppo_lag_${JOB}_0.out
CFG=expert_ppo_lag_config_${JOB}_${SEED}.yaml
OUT=docs/results/$TAG

for f in "$LOG" "$CFG" "$PLOT"; do [ -f "$f" ] || { echo "missing file: $f"; exit 1; }; done
[ -d "$RUN" ] || { echo "missing run folder: $RUN"; exit 1; }
compgen -G "$GLOB" > /dev/null || { echo "no folders match: $GLOB"; exit 1; }

mkdir -p "$OUT/figures" "$OUT/data"
cp "$CFG" "$OUT/config_used.yaml"
head -20 "$LOG" > "$OUT/log_header.txt"
[ "$FULL" = "--full-log" ] && gzip -c "$LOG" > "$OUT/train_log.out.gz"

python "$PLOT" --random-ref ${RANDOM_REF:-0.048 0.167} --run-dir "$RUN" --log "$LOG" --eval-glob "$GLOB" \
  --out "$OUT/figures" --csv "$OUT/data" --tag "s${SEED}_${TAG}" --budget "$BUDGET" \
  --caption "30 episodes per checkpoint, 1 seed, deterministic actions"

echo "--- $OUT ---"; du -sh "$OUT"; find "$OUT" -type f | sort