# Multi-Agent ICRL

Extending existing single-agent Inverse Constrained RL methods
(GACL/ICRL/VICRL, from
[ICRL-benchmarks-public](https://github.com/Guiliang/ICRL-benchmarks-public))
to multi-agent environments.

## Structure

```
algos/            constraint-learning algorithm code (ported from the benchmark repo)
envs/             multi-agent env wrappers
experts/          constrained-MARL expert training, for generating demonstrations
configs/          training configs
scripts/          setup and utility scripts
docs/             env screening results, SB3 diff analysis
progress_update/  weekly progress notes
logs/             SLURM job logs
```

## Setup

Three separate venvs are required.

```bash
bash scripts/setup_modern_env.sh /path/to/venv_modern   # DSSE, Box Jump, dilemma envs
bash scripts/setup_pogema_env.sh /path/to/venv_pogema   # POGEMA (needs isolation)
module load python/3.10                                 # Safety-Gymnasium needs Python 3.8-3.10
bash scripts/setup_safety_gym_env.sh /path/to/venv_safety
```
Notes for Alliance:
- `setup_safety_gym_env.sh` works around three issues: the unbuildable `pygame==2.1.0` pin,
  pip rejecting `manylinux` wheels (mujoco 2.3.3 is installed from a renamed wheel), and
  headless rendering. It is safe to rerun.
- Set `export MUJOCO_GL=egl` before importing mujoco, including in sbatch scripts.
- Python 3.11+ is unsupported by Safety-Gymnasium (pygame).

## Screening and probes

Screen a PettingZoo-based or POGEMA environment:
```bash
source /path/to/venv_modern/bin/activate   # or venv_pogema
cd scripts
python screen_env.py --only <dsse|boxjump|pogema|dilemma_mrochk|dilemma_tianyu>
```
Safety-Gymnasium is not part of `screen_env.py`; it has its own probes (run in
`venv_safety` with `MUJOCO_GL=egl`):

| Script | Checks |
|---|---|
| `scripts/test_safety_multigoal.py` | env builds and steps; shape of rewards, costs and info |
| `scripts/test_contact_other.py` | whether `cost_contact_other` fires under random actions |
| `scripts/test_contact_diagnose.py` | identifies the two ants from the MuJoCo model and compares real contacts with the cost key |
| `scripts/test_ant_collision_masks.py` | whether the two ants can physically collide |

```bash
python scripts/test_safety_multigoal.py SafetyAntMultiGoal2-v0 300
python test_contact_other.py SafetyAntMultiGoal2-v0 10
python test_contact_diagnose.py SafetyAntMultiGoal2-v0 3
python test_ant_collision_masks.py SafetyAntMultiGoal2-v0

## Docs

- `docs/env_screening.md` — which multi-agent envs work
- `docs/sb3_diff/README.md` — what the benchmark repo modified in Stable-Baselines3
- `progress_update/` — weekly status


