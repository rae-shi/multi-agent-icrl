# Research log

### 2026-10-26: two-row adapter, first runs

**Goal.** Get every stage (PPO-Lag expert → expert data → ICRL → plots) running on MultiGoal level 1, with one row per agent.
**Setup.** Fir cluster, CPU-only SLURM jobs (`--account=def-s2ganapa`), `venv_safety` (Python 3.10.13, numpy 1.22.4, torch 2.6.0), Safety-Gymnasium `ae966e5`, ICRL code `0d79d2c` plus patches 0001-0006.

### Runs

**1. Symmetry check** (`verify_symmetry.py`, 10 min, 4 GB, 2 CPUs)
- Why: check the condition for a shared policy.
- Result: observation bounds and action spaces identical (8-d actions, 208-d observations). Both agents received identical observations in 50/50 steps. Layout: `[0:40]` unbounded, `[40:104]` in [0,1], `[104:144]` unbounded, `[144:208]` in [0,1].
- Source reading: same reward formula and goal parameters for both agents (agent 0 chases the red goal, agent 1 the blue one), same hazard-cost function on each agent's own position, horizon 1000.

**2. API and cost checks** (`inspect_step.py`, `check_cost_keys.py`)
- `step()` returns 6 items (obs, reward, cost, terminated, truncated, info), each a dict per agent. `contact_other_cost` defaults to 1.0.
- With `contact_other_cost = 0`: 1000 steps, 69 agent-steps with cost > 0, 0 mismatches between the cost item and `cost_sum`, no `cost_contact_other` key.

**3. Two-row adapter, real-environment smoke test** (`scripts/smoke/smoke_test_pair.py`)
- SMOKE TEST PASSED. 4 rows from 2 simulations, per-row monitors match the violations seen step by step, the two rows of a simulation end together, and the patched `evaluate_icrl_policy` and sampler run on 2 rows.
- Numbers from attempt 2: 266 simulation steps/s raw; 435 simulation steps/s over 2 workers (869 rows/s).
- Random policy, 3000 steps: 8 episodes, lengths 69 / 254 / 1000 (min / median / max), 7 of 8 ended early. Violating steps: agent 0 = 144, agent 1 = 502 (violations in 4/8 vs 7/8 episodes).

**4. PPO-Lag debug run, two-row** (`train_policy.py …Pair-v0.yaml -n 2 -s 123 -d 1`): exit 0, 30 iterations in 16.6 min, last evaluation mean return 0.086. Per-row monitors:

| Row | Simulation, agent | Episodes | Steps | Episodes with a violation |
|---|---|---|---|---|
| r0 | 0, agent 0 | 94 | 12,870 | 18 (19%) |
| r1 | 0, agent 1 | 94 | 12,870 | 48 (51%) |
| r2 | 1, agent 0 | 91 | 14,008 | 25 (27%) |
| r3 | 1, agent 1 | 91 | 14,008 | 33 (36%) |

Pooled: agent 0 43/185 (23%), agent 1 81/185 (44%)

**5. Tiny ICRL pipeline** (`tests/test_expert_data_mock.py` → `scripts/pipeline/generate_expert_data.py --run-dir <debug run> --iteration 29 --episodes 5` → `train_icrl.py …Pair-v0.yaml -n 2 -s 123 -d 1`):

| Row| Simulation, agent | Violating/episodes | reward |
|---|---|---| --- |
|r0 |(simulation 0, agent 0) |	29/97	|    -0.032 |
|r1 |(simulation 0, agent 1) |	44/97	|	-0.011  |
|r2 |(simulation 1, agent 0) |	30/100	|	-0.194  |
|r3 |(simulation 1, agent 1) |	62/100	|	0.010   |

### Code reading (no runs)
- PPO-Lag updates its multiplier from the mean of the original (unnormalized) cost over the whole rollout buffer, so the budget is a fraction of violating steps, pooled over all rows (both agents).
- `VecNormalizeWithCost` rescales `info['cost']`; the raw 0/1 flag stays in `lag_cost`.