"""
Env screening harness for the MACI project.

Encodes the conformance checks used in docs/env_screening.md so new
candidates can be screened the same way. Each candidate gets its own
function returning a small dict of results; add new candidates by
following the same pattern.

Usage:
    python screen_env.py                # run all registered candidates
    python screen_env.py --only pogema  # run just one

Note: some candidates require separate venvs due to conflicting pins
(see docs/env_screening.md, "Dependency conflict, confirmed"). This
script assumes whatever's importable in the current environment; run it
once per venv if you're screening across incompatible env families.
"""
import argparse
import sys
import traceback
from pathlib import Path

# Universal, self-locating default: env_clones lives alongside this script
# (created there by setup_modern_env.sh), regardless of where you run
# `python screen_env.py` from or what venv/absolute paths are in play.
SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CLONE_DIR = SCRIPT_DIR / "env_clones"


def result(name, installs, conformance, notes, info_keys=None):
    return {
        "name": name,
        "installs": installs,
        "conformance": conformance,
        "info_keys": info_keys or [],
        "notes": notes,
    }


def screen_pogema():
    name = "POGEMA"
    try:
        from pettingzoo.test import parallel_api_test
        from pogema.integrations.pettingzoo import parallel_env
        from pogema import GridConfig
    except Exception as e:
        return result(name, False, False, f"import failed: {e!r}")

    try:
        env = parallel_env(GridConfig(num_agents=4, size=8, density=0.3, seed=0))
        parallel_api_test(env, num_cycles=50)
    except Exception as e:
        return result(name, True, False, f"conformance failed: {e!r}")

    # check info dict content
    env = parallel_env(GridConfig(num_agents=4, size=8, density=0.3, seed=0))
    obs, infos = env.reset(seed=0)
    actions = {a: env.action_space(a).sample() for a in env.agents}
    obs, rews, terms, truncs, infos = env.step(actions)
    sample_info = next(iter(infos.values()), {})

    return result(
        name, True, True,
        "no built-in cost/collision signal exposed by default; "
        "GridConfig(collision_system=...) tracks collisions internally "
        "but needs a custom wrapper to surface it",
        info_keys=list(sample_info.keys()),
    )


def screen_pettingzoo_dilemma_envs_tianyu(repo_path):
    """Requires: git clone https://github.com/tianyu-z/pettingzoo_dilemma_envs.git
    and repo_path pointing at it, added to sys.path before calling."""
    name = "PettingZoo Dilemma Envs (tianyu-z)"
    sys.path.insert(0, repo_path)
    try:
        from pettingzoo.test import api_test
        import dilemma_pettingzoo as dp
    except Exception as e:
        return result(name, False, False, f"import failed: {e!r}")

    try:
        env = dp.env(game="pd")
        api_test(env, num_cycles=50)
    except Exception as e:
        return result(
            name, True, False,
            f"conformance failed: {e!r} -- if this is the agent_selector "
            "TypeError, patch the import (see setup_modern_env.sh). If "
            "it's something else (e.g. 'Out of bounds observation'), "
            "that's a SEPARATE, deeper bug in this repo's own game logic, "
            "confirmed present even after the agent_selector patch is "
            "applied -- not yet root-caused further",
        )
    return result(name, True, True, "OK")


def screen_pettingzoo_dilemmas_mrochk(repo_path):
    """Requires: git clone https://github.com/mrochk/pettingzoo-dilemmas.git
    and repo_path pointing at it, added to sys.path before calling.
    Note: pip install of this package produces an empty wheel (packaging
    bug in pyproject.toml) -- use PYTHONPATH / sys.path, not pip, until
    that's fixed upstream or patched locally."""
    name = "PettingZoo Dilemmas (mrochk)"
    sys.path.insert(0, repo_path)
    try:
        from pettingzoo.test import parallel_api_test
        import enum
        from pettingzoo_dilemmas import matrix_game_v0
    except Exception as e:
        return result(name, False, False, f"import failed: {e!r}")

    try:
        class Moves(enum.Enum):
            A = 0
            B = 1

        reward_matrix = {
            (Moves.A, Moves.A): (3, 3),
            (Moves.A, Moves.B): (0, 5),
            (Moves.B, Moves.A): (5, 0),
            (Moves.B, Moves.B): (1, 1),
        }
        env = matrix_game_v0.env(Moves=Moves, reward_matrix=reward_matrix, nrounds=50)
        parallel_api_test(env, num_cycles=50)
    except Exception as e:
        return result(name, True, False, f"conformance failed: {e!r}")

    return result(
        name, True, True,
        "passes as Parallel API only, despite 'env' naming convention "
        "suggesting AEC; ground-truth reward matrix is user-defined, "
        "good fit for FP-MACI validation against exact equilibria",
    )


def screen_cookingzoo():
    name = "CookingZoo"
    try:
        from cooking_zoo.environment.cooking_env import env as cz_env
    except Exception as e:
        return result(name, False, False, f"import failed: {e!r}")

    try:
        e = cz_env(level="coop_test", meta_file="example", num_agents=2,
                   max_steps=100, recipes=["TomatoLettuceSalad"])
    except Exception as ex:
        return result(
            name, True, False,
            f"instantiation failed: {ex!r} -- level assets not bundled "
            "in the pip/git-installed package; run from a full repo clone "
            "so relative asset paths resolve, or copy the asset directory "
            "into site-packages manually",
        )
    return result(name, True, True, "OK")


def screen_boxjump():
    """Requires system 'swig' binary + box2d-py before pip install ./boxjump."""
    name = "Box Jump"
    try:
        from pettingzoo.test import parallel_api_test
        from boxjump.box_env import BoxJumpEnvironment
    except Exception as e:
        return result(name, False, False, f"import failed: {e!r} -- needs system "
                      "'swig' binary (apt/brew install swig) before box2d-py builds")

    try:
        env = BoxJumpEnvironment(num_boxes=4, render_mode=None)
        parallel_api_test(env, num_cycles=50)
    except Exception as e:
        return result(name, True, False, f"conformance failed: {e!r}")

    env = BoxJumpEnvironment(num_boxes=4, render_mode=None)
    obs, infos = env.reset(seed=0)
    return result(
        name, True, True,
        "empty info dicts, no native cost signal -- but obs already includes "
        "raycast distances to neighboring boxes, usable to derive a proximity "
        "cost externally",
        info_keys=list(next(iter(infos.values()), {}).keys()),
    )


def screen_dsse():
    name = "DSSE"
    try:
        from pettingzoo.test import parallel_api_test
        from DSSE import DroneSwarmSearch
    except Exception as e:
        return result(name, False, False, f"import failed: {e!r}")

    try:
        env = DroneSwarmSearch(grid_size=15, drone_amount=3, person_amount=2,
                                timestep_limit=50)
        parallel_api_test(env, num_cycles=30)
    except Exception as e:
        return result(name, True, False, f"conformance failed: {e!r}")

    return result(
        name, True, True,
        "partial signal: constants.py defines a 'drones_collision' reward "
        "field (currently unused, default 0), and env.py has a commented-out "
        "call to compute_drone_collision(...). Checked git history: this "
        "method existed before commit 480e5e1 and has since been fully "
        "removed from the active codebase -- would need porting/rewriting "
        "against the current env_base.py structure, not just re-enabling",
        info_keys=["Found"],
    )


def screen_crazyrl():
    name = "Crazy-RL"
    try:
        import crazy_rl  # noqa: F401
    except Exception as e:
        return result(
            name, False, False,
            f"import failed: {e!r} -- pins torch<2.0 and needs Python<3.12; "
            "both stale, install in a dedicated older-Python venv if pursuing",
        )
    return result(name, True, None, "installed but not further tested")


def screen_racecar_gym():
    name = "Racecar Gym"
    try:
        import racecar_gym  # noqa: F401
    except Exception as e:
        return result(
            name, False, False,
            f"import failed: {e!r} -- repo's build depends on distutils, "
            "removed from stdlib in Python 3.12; retest with Python <=3.11 "
            "(e.g. Alliance's `module load python/3.10.13`) before ruling out",
        )
    return result(name, True, None, "installed but not further tested")


CANDIDATES = {
    "pogema": lambda args: screen_pogema(),
    "dilemma_tianyu": lambda args: screen_pettingzoo_dilemma_envs_tianyu(args.dilemma_tianyu_path),
    "dilemma_mrochk": lambda args: screen_pettingzoo_dilemmas_mrochk(args.dilemma_mrochk_path),
    "cookingzoo": lambda args: screen_cookingzoo(),
    "boxjump": lambda args: screen_boxjump(),
    "dsse": lambda args: screen_dsse(),
    "crazyrl": lambda args: screen_crazyrl(),
    "racecar_gym": lambda args: screen_racecar_gym(),
}


def print_result(r):
    status = "PASS" if r["conformance"] else ("INSTALL-ONLY" if r["installs"] else "FAIL")
    print(f"\n=== {r['name']} :: {status} ===")
    print(f"  installs cleanly:   {r['installs']}")
    print(f"  passes conformance: {r['conformance']}")
    if r["info_keys"]:
        print(f"  info dict keys:     {r['info_keys']}")
    print(f"  notes: {r['notes']}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--only", choices=list(CANDIDATES.keys()), default=None)
    parser.add_argument("--dilemma-tianyu-path", dest="dilemma_tianyu_path",
                         default=str(DEFAULT_CLONE_DIR / "pettingzoo_dilemma_envs"))
    parser.add_argument("--dilemma-mrochk-path", dest="dilemma_mrochk_path",
                         default=str(DEFAULT_CLONE_DIR / "pettingzoo-dilemmas"))
    args = parser.parse_args()

    to_run = [args.only] if args.only else list(CANDIDATES.keys())
    for key in to_run:
        try:
            r = CANDIDATES[key](args)
        except Exception:
            print(f"\n=== {key} :: CRASHED ===")
            traceback.print_exc()
            continue
        print_result(r)