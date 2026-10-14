# Provenance

Code in this directory is copied from
https://github.com/Guiliang/ICRL-benchmarks-public (BSD 3-Clause, see `LICENSE`),
the code of Liu et al., "Benchmarking Constraint Inference in Inverse Reinforcement
Learning", ICLR 2023.

- Upstream commit: 0d79d2c5bdf8e5fe87f89626036a423a806c4c90 (committed 2023-11-10)
- Copied on: 2026-10-14
- Not copied: `.git`, `data/`, `save_model/`, `commonroad_environment/`, `images&others/`.

## Known issues seen with the original setup (not applied; fix only if they occur)

- numpy 1.24 removed the `np.bool` alias. With a newer numpy this raises an
  AttributeError in `stable_baselines3/common/base_class.py` (about line 348) and
  `stable_baselines3/common/vec_env/dummy_vec_env.py` (about line 33); change both to
  `np.bool_`. Our venv uses numpy 1.22.4, where the alias still works.
- `constraint_models/constraint_net/gail_net.py`, `reshape_actions`: actions with more
  than two dimensions are not flattened. Seen with the HalfCheetah GAIL run on multi-env
  data. The fix was adding `elif len(acs.shape) > 2: acs, _ = self.flatten(acs)` after the
  discrete-actions branch.
- `config/mujuco_BlockedHalfCheetah/train_GAIL_HCWithPos-v0.yaml` has
  `device: 'cuda'`; the earlier cluster runs used `cpu`. Set `device` explicitly in
  every new config.

Later changes made in this repository are in its git history.

## Algorithm names: paper vs code

Status: hypothesis from names, configs and `gail_net.py`; not yet verified in the code.

| Paper | Code name | Evidence |
|---|---|---|
| GACL | GAIL | `GailDiscriminator.cost_function` returns 1 - D(obs, acs); GAIL configs set `learn_cost: true` |
| BC2L | Binary | name match |
| MECL | ICRL | by elimination (baseline list: BC2L, GACL, ICRL, VICRL) |
| VICRL | VICRL | name match |
| PPO-Lag | PPO-Lag | name match |
| (reference) | PPO | unconstrained baseline |

## Running

```bash
export PYTHONPATH="$PWD/algos/icrl:$PYTHONPATH"
```

`algos/icrl/stable_baselines3` is a modified SB3 0.9.0. Because it is first on
PYTHONPATH it shadows any pip-installed stable-baselines3.
