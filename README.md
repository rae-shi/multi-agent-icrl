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

Two separate venvs are required.

```bash
bash scripts/setup_modern_env.sh /path/to/venv_modern   # DSSE, Box Jump, dilemma envs
bash scripts/setup_pogema_env.sh /path/to/venv_pogema   # POGEMA (needs isolation)
```

Then screen any environment:

```bash
source /path/to/venv_modern/bin/activate   # or venv_pogema
cd scripts
python screen_env.py --only <dsse|boxjump|pogema|dilemma_mrochk|dilemma_tianyu>
```

## Docs

- `docs/env_screening.md` — which multi-agent envs work
- `docs/sb3_diff/README.md` — what the benchmark repo modified in Stable-Baselines3
- `progress_update/` — weekly status