#!/bin/bash
# ==========================================================================
# POGEMA needs its OWN fully isolated venv since PettingZoo itself declares
# gymnasium>=1.0.0, so pip's resolver upgrades gymnasium the moment ANY
# other pettingzoo-dependent package (DSSE, boxjump, dilemma envs) is
# installed in the same venv, silently breaking POGEMA at runtime.
#
# Do NOT install anything else into this venv beyond what's below.
#
# Usage:
#   bash setup_pogema_env.sh /path/to/new/venv
# ==========================================================================
set -e

VENV_DIR="${1:?Usage: bash setup_pogema_env.sh /path/to/venv}"
 
module load python/3.12.4
 
python -m venv "$VENV_DIR"
source "$VENV_DIR/bin/activate"
pip install --upgrade pip setuptools wheel
 
echo "== Installing pettingzoo + pogema (isolated) =="

pip install --quiet pettingzoo==1.27.0
pip install --quiet pogema
 
echo ""
echo "== Verifying gymnasium pin held =="
pip show gymnasium | grep Version
echo "(expect 0.28.1 -- if it's anything else, something upgraded it and"
echo " POGEMA will silently fail at runtime, not at install time)"
 
echo ""
echo "== Setup done. Venv: $VENV_DIR =="
echo "Now run: python screen_env.py --only pogema"