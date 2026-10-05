#!/usr/bin/env python3
"""
test_safety_multigoal.py  (adapts to PettingZoo-style per-agent API)

Probe Safety-Gymnasium's Safe Multi-Agent envs. Findings from the first run:
reset() returns (obs_dict, info_dict) keyed by agent id, and observation_space /
action_space are METHODS (called with an agent id), not attributes.

  * derives agent ids from the reset() observation dict,
  * builds a dict of per-agent random actions,
  * accepts either a 6-tuple step (obs, rew, cost, term, trunc, info) or a
    5-tuple PettingZoo step (cost then looked for inside the info dicts),
  * prints structure so we can see exactly where cost lives and whether any
    collision-related key exists.

Usage (in venv_safety, MUJOCO_GL=egl exported):
    python test_safety_multigoal.py [ENV_ID] [N_STEPS]
"""
import sys
from collections import defaultdict
import numpy as np
import safety_gymnasium

env_id = sys.argv[1] if len(sys.argv) > 1 else "SafetyAntMultiGoal1-v0"
n_steps = int(sys.argv[2]) if len(sys.argv) > 2 else 300


def describe(x, depth=0, name="value"):
    pad = "  " * depth
    if isinstance(x, dict):
        print(f"{pad}{name}: dict with keys {list(x.keys())}")
        for k, v in list(x.items())[:6]:
            describe(v, depth + 1, str(k))
    elif isinstance(x, (list, tuple)):
        print(f"{pad}{name}: {type(x).__name__} of len {len(x)}")
        for i, v in enumerate(x[:4]):
            describe(v, depth + 1, f"[{i}]")
    elif isinstance(x, np.ndarray):
        print(f"{pad}{name}: ndarray shape={x.shape} dtype={x.dtype}")
    elif hasattr(x, "shape") and hasattr(x, "sample"):
        print(f"{pad}{name}: {type(x).__name__} shape={x.shape}")
    else:
        print(f"{pad}{name}: {type(x).__name__} = {x}")


def get_space(env, name, agent):
    attr = getattr(env, name)
    return attr(agent) if callable(attr) else attr


def as_float(x):
    try:
        return float(np.sum(x))
    except Exception:
        return 0.0


def any_true(x):
    if isinstance(x, dict):
        return any(bool(np.any(v)) for v in x.values())
    return bool(np.any(x))


def cost_from_info(infos):
    """Pull any entries whose key contains 'cost' out of per-agent info dicts."""
    per_agent = {}
    if isinstance(infos, dict):
        for agent, inf in infos.items():
            if isinstance(inf, dict):
                per_agent[agent] = sum(as_float(v) for k, v in inf.items() if "cost" in str(k).lower())
    return per_agent


print(f"=== Making {env_id} ===")
env = safety_gymnasium.make(env_id)
print(f"env type: {type(env)}")

print("\n=== reset() ===")
reset_out = env.reset()
obs, info = reset_out if isinstance(reset_out, tuple) and len(reset_out) == 2 else (reset_out, {})
describe(reset_out, name="reset output")
agents = list(obs.keys()) if isinstance(obs, dict) else ["agent_0"]
print(f"agents: {agents}")

print("\n=== Per-agent spaces ===")
for a in agents:
    describe(get_space(env, "observation_space", a), name=f"observation_space({a})")
    describe(get_space(env, "action_space", a), name=f"action_space({a})")


def random_actions():
    return {a: get_space(env, "action_space", a).sample() for a in agents}


print("\n=== One step ===")
out = env.step(random_actions())
print(f"step returned {len(out)} items")
for i, item in enumerate(out):
    describe(item, name=f"out[{i}]")

print(f"\n=== {n_steps} random steps: cost statistics ===")
env.reset()
total_cost = []
per_agent_cost = defaultdict(float)
info_keys = set()
for t in range(n_steps):
    out = env.step(random_actions())
    if len(out) == 6:
        _, _, cost, term, trunc, infos = out
        if isinstance(cost, dict):
            for a, c in cost.items():
                per_agent_cost[a] += as_float(c)
            total_cost.append(sum(as_float(c) for c in cost.values()))
        else:
            total_cost.append(as_float(cost))
    elif len(out) == 5:
        _, _, term, trunc, infos = out
        pc = cost_from_info(infos)
        for a, c in pc.items():
            per_agent_cost[a] += c
        total_cost.append(sum(pc.values()))
    else:
        print(f"Unexpected step length {len(out)}; stopping.")
        break
    if isinstance(infos, dict):
        for inf in infos.values():
            if isinstance(inf, dict):
                info_keys.update(inf.keys())
    if any_true(term) or any_true(trunc):
        env.reset()

costs = np.array(total_cost)
if len(costs):
    print(f"steps with total cost>0: {(costs > 0).sum()} / {len(costs)}  "
          f"(mean {costs.mean():.4f}, max {costs.max():.2f})")
print(f"per-agent cumulative cost: {dict(per_agent_cost)}")
print(f"info keys seen: {sorted(map(str, info_keys))}")
print("\nLook for collision-related keys or per-agent cost entries above.")