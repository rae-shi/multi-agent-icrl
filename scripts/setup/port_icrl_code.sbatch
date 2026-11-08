#!/bin/bash
#SBATCH --account=def-s2ganapa
#SBATCH --time=00:30:00
#SBATCH --cpus-per-task=1
#SBATCH --mem=4G
#SBATCH --job-name=port_icrl
#SBATCH --output=%x-%j.out
# Submit with:  sbatch scripts/port_icrl_code.sbatch
# If submission asks for a partition, add a --partition line for a short CPU partition.
set -euo pipefail

cd "$HOME/projects/def-s2ganapa/raeshi/multi-agent-icrl"
bash scripts/port_icrl_code.sh
