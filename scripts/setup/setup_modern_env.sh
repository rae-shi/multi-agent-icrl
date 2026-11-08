#!/bin/bash
# ==========================================================================
# Sets up the "modern" venv: everything that's fine with gymnasium>=1.0.0.
# This covers DSSE, Box Jump, and pettingzoo-dilemmas (mrochk).
#
# POGEMA is DELIBERATELY NOT included here. See setup_pogema_env.sh.
# Root cause: POGEMA pins gymnasium==0.28.1. Any package that lists
# `pettingzoo` as a dependency (DSSE, boxjump, dilemmas, not just POGEMA)
# makes pip's resolver "fix" gymnasium upward to satisfy PettingZoo's own
# declared gymnasium>=1.0.0 requirement, even when you're not touching
# PettingZoo directly in that install command. This SILENTLY breaks POGEMA
# at runtime (not a loud install-time error).
#
# Run this on the Alliance LOGIN node (needs internet access).
# Do not run this inside a SLURM job.
#
# Usage:
#   bash setup_modern_env.sh /path/to/new/venv
# ==========================================================================
set -e

# Self-locating: env_clones always lives alongside this script, regardless
# of where the venv itself is created or what directory you run this from.
# This is what makes the paths "universal" -- no absolute path or shell
# variable needs to be remembered or retyped later.
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLONE_DIR="$SCRIPT_DIR/env_clones"

VENV_DIR="${1:?Usage: bash setup_modern_env.sh /path/to/venv}"

module load python/3.12.4
module load swig/4.2.1

echo "== Creating venv at $VENV_DIR =="
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"

# NOTE: setuptools/wheel are NOT bundled by venv in Python 3.12+ the way
# they were in older Pythons. Without this line, DSSE fails to build with
# "Cannot import 'setuptools.build_meta'".
pip install --upgrade pip setuptools wheel

echo "== Installing pettingzoo, DSSE, Box Jump deps =="
# NOTE: pinning pettingzoo explicitly to the version this screening was
# actually validated against. Alliance's wheelhouse carries pettingzoo
# 1.23.1 (confirmed via `avail_wheels pettingzoo`), which is meaningfully
# different from 1.27.0 (PyPI latest, what every result in
# docs/env_screening.md was tested against), notably closer to what the
# dilemma-envs repos target (1.22.3), so bugs like the agent_selector
# rename may not even reproduce on 1.23.1. Pinning avoids silently testing
# against an unvalidated version if pip is configured to prefer the local
# wheelhouse by default.
pip install --quiet pettingzoo==1.27.0 DSSE box2d-py pygame

mkdir -p "$CLONE_DIR"
cd "$CLONE_DIR"

echo "== Box Jump =="
if [ ! -d "$CLONE_DIR/boxjump" ]; then
    git clone --quiet https://github.com/zzbuzzard/boxjump.git
fi
pip install --quiet ./boxjump

echo "== PettingZoo Dilemmas (mrochk) =="
if [ ! -d "$CLONE_DIR/pettingzoo-dilemmas" ]; then
    git clone --quiet https://github.com/mrochk/pettingzoo-dilemmas.git
fi
# NOTE: this repo's pyproject.toml has a packaging bug: declares
# `packages = ["dilemmas"]` but the real source dir is `pettingzoo_dilemmas/`,
# so `pip install` succeeds but installs an EMPTY package. Do not pip
# install it, use PYTHONPATH instead. screen_env.py's
# --dilemma-mrochk-path flag handles this for you.

echo "== PettingZoo Dilemma Envs (tianyu-z) =="
if [ ! -d "$CLONE_DIR/pettingzoo_dilemma_envs" ]; then
    git clone --quiet https://github.com/tianyu-z/pettingzoo_dilemma_envs.git
fi
# NOTE: as-is, this fails against modern pettingzoo:
#   TypeError: 'module' object is not callable
# Cause: `from pettingzoo.utils import agent_selector` used as a callable;
# renamed to AgentSelector in current pettingzoo. Patching it here:
DILEMMA_FILE="$CLONE_DIR/pettingzoo_dilemma_envs/dilemma_pettingzoo.py"
if grep -q "from pettingzoo.utils import agent_selector" "$DILEMMA_FILE" 2>/dev/null; then
    sed -i "s/from pettingzoo.utils import agent_selector/from pettingzoo.utils import AgentSelector as agent_selector/" "$DILEMMA_FILE"
    echo "  patched agent_selector -> AgentSelector"
fi

echo ""
echo "== Setup done. Venv: $VENV_DIR =="
echo "Covers: pettingzoo, DSSE, Box Jump, both dilemma env repos."
echo "Env repos cloned into: $CLONE_DIR"
echo ""
echo "POGEMA needs its OWN separate venv -- run:"
echo "    bash setup_pogema_env.sh ${VENV_DIR}_pogema"
echo ""
echo "Now run (paths auto-detected, no need to pass --dilemma-*-path):"
echo "  python screen_env.py --only dsse"
echo "  python screen_env.py --only boxjump"
echo "  python screen_env.py --only dilemma_mrochk"
echo "  python screen_env.py --only dilemma_tianyu"