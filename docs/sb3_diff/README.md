# SB3 Modifications: ICRL-benchmarks-public vs. upstream v0.9.0

`sb3_changed_files.txt` and `sb3_modifications.patch` in this folder are the
raw diff between the `stable_baselines3` bundled inside
[ICRL-benchmarks-public](https://github.com/Guiliang/ICRL-benchmarks-public)
and vanilla upstream SB3 `v0.9.0` (the version declared in the bundled
copy's own `version.txt`, the closest-matching tag before diffing).

This file is about what the **original benchmark authors** changed in SB3
itself to support constrained RL, before we ever touched the repo.

## Why this matters for porting

Before porting GACL/ICRL/VICRL to a modern SB3, we need to know exactly
which parts of SB3's internals were modified to carry a constraint/cost
signal through training and these modifications have to be re-implemented
against the new SB3 version. Losing one of these wouldn't crash training,
it would just quietly produce wrong numbers.

## Categorization

| File | Type | Category | What it does |
|---|---|---|---|
| `common/dual_variable.py` | new | **Constraint-related** | `DualVariable`, `PIDLagrangian` — the Lagrangian multiplier machinery, core to PPO-Lag |
| `common/vec_env/vec_cost_wrapper.py` | new | **Constraint-related** | `VecCostWrapper` — computes a per-step cost from a pluggable `cost_function`, injects it into `info['cost']` |
| `common/buffers.py` | modified | **Constraint-related** | Adds `CustomRolloutBuffer`, `RolloutBufferWithCost` — parallel reward/cost GAE, returns, advantages, values |
| `common/type_aliases.py` | modified | **Constraint-related** | Adds `RolloutBufferWithCostSamples`, the named tuple the buffer above returns |
| `common/monitor.py` | modified | **Constraint-related, and hardcoded — see below** | Adds `track_keywords` for info-dict tracking, and a hardcoded HalfCheetah-specific constraint check |
| `stable_baselines3/ppo_lag/` (3 files, all new) | new | **Constraint-related** | Full PPO-Lagrangian implementation |
| `stable_baselines3/iteration/policy_interation_lag.py` | new | **Constraint-related** | Lagrangian policy iteration variant |
| `stable_baselines3/iteration/policy_interation_gail.py` | new | **Constraint-related** | GAIL policy iteration variant |
| `common/vec_env/subproc_vec_env.py`, `dummy_vec_env.py`, `vec_normalize.py`, `vec_normalize_fixed.py` (new), `vec_transpose.py`, `vec_env/__init__.py` | modified/new | **Constraint-related** | VecEnv plumbing so cost travels alongside obs/reward through the vectorized-env stack |
| `common/on_policy_algorithm.py` | modified | Constraint-related (**not yet inspected in detail**) | Presumed rollout-collection loop populating the cost buffer above — needs a direct look before porting |
| `common/callbacks.py` | modified | Likely constraint-related (**not yet inspected**) | Presumed cost/violation metric logging |
| `common/logger.py` | modified | Likely constraint-related (**not yet inspected**) | Presumed source of the `reward_nc`/`constraint` monitor.csv columns |
| `common/results_plotter.py` | modified | Likely constraint-related (**not yet inspected**) | Presumed cost-alongside-reward plotting |
| `common/evaluation.py` | modified | Likely constraint-related (**not yet inspected**) | Presumed violation reporting during eval |
| `common/base_class.py` | modified | Mixed | Partly a `np.bool`→`np.bool_` NumPy-deprecation fix (incidental), partly presumed cost-buffer/algorithm wiring (not yet isolated) |
| `common/torch_layers.py`, `common/policies.py`, `a2c/policies.py`, `dqn/policies.py`, `ppo/policies.py`, `sac/policies.py`, `td3/policies.py` | modified | Likely incidental (**not yet inspected**) | Presumed minor API-compatibility tweaks, not constraint-specific |
| `common/utils.py`, `common/bit_flipping_env.py`, `dqn/dqn.py`, `__init__.py`, `LICENSE` | modified | Incidental | Version bump / minor unrelated changes |

Rows marked "not yet inspected in detail" were categorized by filename and
context, not by reading the actual diff content line by line, flagged here.
