#!/usr/bin/env python3
"""
test_contact_other.py

Does `cost_contact_other` (set by contact_other_cost=1.0 in MultiGoal levels 1/2)
actually fire, for which agents, and how far apart are the agents when it does?

Costs are read from info['agent_0'|'agent_1'] (the shared nested cost dicts) with
.get(), because the key only exists on steps where a matching contact occurs.

Usage (venv_safety, MUJOCO_GL=egl):
    python test_contact_other.py [ENV_ID] [N_EPISODES]
"""
import sys
import numpy as np
import safety_gymnasium

env_id = sys.argv[1] if len(sys.argv) > 1 else "SafetyAntMultiGoal2-v0"
n_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 10

env = safety_gymnasium.make(env_id)
agents = ["agent_0", "agent_1"]


def agent_dist(env):
    """Planar distance between the two agents; None if the attribute path differs."""
    try:
        task = env.unwrapped.task
        p0 = np.asarray(task.agent.pos_0, dtype=float)[:2]
        p1 = np.asarray(task.agent.pos_1, dtype=float)[:2]
        return float(np.linalg.norm(p0 - p1))
    except Exception:
        return None


steps = 0
events = {a: 0 for a in agents}
both = 0
dist_at_contact, dist_all = [], []
total_cost_steps = 0

for ep in range(n_eps):
    env.reset()
    while True:
        actions = {a: env.action_space(a).sample() for a in agents}
        _, _, costs, term, trunc, infos = env.step(actions)
        steps += 1
        info = infos["agent_0"]  # same shared dict for both agents
        hit = {a: (info.get(a, {}) or {}).get("cost_contact_other") for a in agents}
        d = agent_dist(env)
        if d is not None:
            dist_all.append(d)
        fired = [a for a in agents if hit[a]]
        for a in fired:
            events[a] += 1
        if len(fired) == 2:
            both += 1
        if fired and d is not None:
            dist_at_contact.append(d)
        if sum(costs.values()) > 0:
            total_cost_steps += 1
        if any(term.values()) or any(trunc.values()):
            break

print(f"env: {env_id} | episodes: {n_eps} | steps: {steps}")
print(f"steps with any cost > 0: {total_cost_steps}")
print(f"cost_contact_other events: {events}  | both agents on same step: {both}")
if dist_all:
    print(f"agent-agent distance over run: min {min(dist_all):.3f}, "
          f"median {np.median(dist_all):.3f}, max {max(dist_all):.3f}")
else:
    print("could not read agent positions via env.unwrapped.task.agent.pos_0/pos_1")
if dist_at_contact:
    print(f"distance at contact steps: min {min(dist_at_contact):.3f}, "
          f"median {np.median(dist_at_contact):.3f}, n={len(dist_at_contact)}")