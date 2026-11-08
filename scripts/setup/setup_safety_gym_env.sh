#!/bin/bash
# setup_safety_gym_env.sh
# Isolated venv for Safety-Gymnasium (incl. Safe Multi-Agent), installed from SOURCE.
#
# Why source: v1.1.0+ is not on PyPI (README). Why Python 3.8-3.10: README says 3.11
# is unsupported (pygame). Alliance has no 3.8, so 3.10 is the practical choice.
#
# Cluster-specific workarounds baked in (all hit and confirmed on Alliance):
#  1. pygame==2.1.0 cannot build from source (no libpng/libjpeg/portmidi headers)
#     -> relax the pin in pyproject.toml so pip picks a prebuilt wheel.
#  2. Alliance's pip only accepts linux_x86_64 wheel tags (no manylinux), so pip
#     falls back to compiling mujoco 2.3.3 and demands MUJOCO_PATH
#     -> download the manylinux wheel with --platform, rename tag, install directly.
#
# Usage:
#   module load python/3.10
#   bash setup_safety_gym_env.sh /path/to/venv_safety
# Env overrides: MUJOCO_VERSION (default 2.3.3), SKIP_PYGAME_PIN=0 to leave pin alone.

set -e

if [ -z "$1" ]; then
    echo "Usage: bash setup_safety_gym_env.sh /path/to/venv_safety"
    exit 1
fi

VENV_PATH="$1"
MUJOCO_VERSION="${MUJOCO_VERSION:-2.3.3}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CLONES_DIR="$SCRIPT_DIR/env_clones"
SG_DIR="$CLONES_DIR/safety-gymnasium"
mkdir -p "$CLONES_DIR"

export MUJOCO_GL="${MUJOCO_GL:-egl}"
export PYOPENGL_PLATFORM="${PYOPENGL_PLATFORM:-egl}"

echo "=== Checking Python version ==="
python3 --version
PYVER=$(python3 -c 'import sys; print(f"{sys.version_info.major}.{sys.version_info.minor}")')
case "$PYVER" in
    3.8|3.9|3.10) echo "OK: Python $PYVER" ;;
    *) echo "ERROR: Python $PYVER unsupported (README: 3.11 breaks pygame). Load python/3.10."; exit 1 ;;
esac
PYTAG="cp${PYVER//./}"

echo "=== Creating venv at $VENV_PATH ==="
if [ ! -d "$VENV_PATH" ]; then
    python3 -m venv "$VENV_PATH"
fi
source "$VENV_PATH/bin/activate"
pip install --upgrade pip setuptools wheel

echo "=== Cloning safety-gymnasium (source install) ==="
if [ ! -d "$SG_DIR" ]; then
    git clone https://github.com/PKU-Alignment/safety-gymnasium.git "$SG_DIR"
fi
cd "$SG_DIR"

if [ "${SKIP_PYGAME_PIN:-1}" = "1" ]; then
    echo "=== Relaxing pygame pin ==="
    FILES=$(grep -rlE "pygame *(==|~=|<=|>=|<|>)" setup.py setup.cfg pyproject.toml requirements*.txt 2>/dev/null || true)
    if [ -z "$FILES" ]; then
        echo "No pygame pin found (already relaxed, or pin lives elsewhere)."
    else
        for f in $FILES; do
            echo "Patching $f"
            [ -f "$f.orig" ] || cp "$f" "$f.orig"
            sed -i -E 's/pygame *(==|~=|<=|>=|<|>) *[0-9][0-9A-Za-z.*]*(, *(<|>|<=|>=|!=) *[0-9][0-9A-Za-z.*]*)*/pygame/g' "$f"
            diff "$f.orig" "$f" || true
        done
    fi
fi

echo "=== Installing mujoco $MUJOCO_VERSION via wheel workaround ==="
if python -c "import mujoco,sys; sys.exit(0 if mujoco.__version__=='$MUJOCO_VERSION' else 1)" 2>/dev/null; then
    echo "mujoco $MUJOCO_VERSION already importable, skipping."
else
    WHEEL_DIR="$(mktemp -d)"
    pip download "mujoco==$MUJOCO_VERSION" --no-deps --only-binary=:all: \
        --platform manylinux2014_x86_64 --python-version "$PYVER" \
        --implementation cp --abi "$PYTAG" -d "$WHEEL_DIR"
    for w in "$WHEEL_DIR"/mujoco-*.whl; do
        new="${w/-manylinux*.whl/-linux_x86_64.whl}"
        mv "$w" "$new"
        pip install --no-deps "$new"
    done
    pip install absl-py glfw pyopengl numpy
    python -c "import mujoco; print('mujoco version:', mujoco.__version__)"
fi

echo "=== Installing safety-gymnasium (editable) ==="
pip install -e .

echo "=== Installed versions ==="
pip list 2>/dev/null | grep -iE "safety|mujoco|gymnasium|pygame" || true

echo "=== Where is pygame referenced in the package? ==="
grep -rn "pygame" safety_gymnasium/ 2>/dev/null | head -20 || true

echo "=== Multi-agent entry points / registrations ==="
grep -rn "make_ma\|MultiGoal" safety_gymnasium/__init__.py 2>/dev/null | head -20 || true
ls safety_gymnasium/tasks 2>/dev/null || true

echo "=== Sanity check: SafetyAntMultiGoal1-v0 ==="
python3 - << 'PY'
import safety_gymnasium
env_id = "SafetyAntMultiGoal1-v0"
ctors = [("make", safety_gymnasium.make)]
if hasattr(safety_gymnasium, "make_ma"):
    ctors.append(("make_ma", safety_gymnasium.make_ma))
for name, ctor in ctors:
    try:
        env = ctor(env_id)
        reset_out = env.reset()
        out = env.step(env.action_space.sample())
        print(f"OK via {name}: step returned {len(out)} items")
        print(str(out)[:600])
        break
    except Exception as e:
        print(f"{name} FAILED: {type(e).__name__}: {e}")
else:
    print("All constructors failed; check MultiGoal docs for the exact call.")
PY

echo "=== Done. Activate: source $VENV_PATH/bin/activate ; export MUJOCO_GL=egl ==="