#!/usr/bin/env python3
"""
test_contact_diagnose.py  (ants identified from the MuJoCo model, not from body_info)

Previous test reused task.agent.body_info[i].geom_names for its own "real contact" check, so it
could not distinguish a cost bug from a bad name list. This test groups geoms by ROOT BODY
(model.body_rootid), picks the two root bodies with the most geoms as the two ants,
and does all contact/teleport tests with those groups.

Reaches into env.unwrapped.task (names read from source, not executed). Each step is
wrapped in try/except.

Usage (venv_safety, MUJOCO_GL=egl):
    python test_contact_diagnose.py [ENV_ID] [N_EPISODES]
"""
import sys
from collections import defaultdict
import numpy as np
import mujoco
import safety_gymnasium

env_id = sys.argv[1] if len(sys.argv) > 1 else "SafetyAntMultiGoal2-v0"
n_eps = int(sys.argv[2]) if len(sys.argv) > 2 else 3
env = safety_gymnasium.make(env_id)
agents = ["agent_0", "agent_1"]
env.reset()
task = env.unwrapped.task
model, data = task.model, task.data


def bname(i):
    return model.body(int(i)).name


def root_of_geom(g):
    return int(model.body_rootid[int(model.geom_bodyid[int(g)])])


groups = defaultdict(list)  # root body id -> geom ids
for g in range(model.ngeom):
    r = root_of_geom(g)
    if r != 0:  # skip world-attached geoms (walls, hazards, pillars...)
        groups[r].append(g)

print("=== Step 1: root bodies by geom count (ants should be the two biggest) ===")
ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
for r, gs in ranked[:6]:
    print(f"root {r:3d} '{bname(r)}': {len(gs)} geoms, e.g. {[model.geom(g).name for g in gs[:3]]}")
print(f"... {len(groups)} non-world root bodies in total")
ant_roots = [r for r, _ in ranked[:2]]
ant_geoms = {r: set(groups[r]) for r in ant_roots}
print(f"ants taken as roots: {[(r, bname(r)) for r in ant_roots]}")

print("\n=== Step 2: which ant do body_info's geom lists belong to? ===")
try:
    for i in (0, 1):
        names = list(task.agent.body_info[i].geom_names)
        owners = [(n, bname(root_of_geom(model.geom(n).id))) for n in names]
        print(f"body_info[{i}].geom_names -> {owners}")
except Exception as e:
    print(f"step 2 failed: {type(e).__name__}: {e}")


def real_contacts():
    a, b = ant_roots
    n = 0
    for c in data.contact[: data.ncon]:
        ra, rb = root_of_geom(c.geom1), root_of_geom(c.geom2)
        if {ra, rb} == {a, b}:
            n += 1
    return n


print("\n=== Step 3: rollout, real ant-ant contacts vs cost_contact_other ===")
try:
    real_steps = key_steps = steps = 0
    min_d = 1e9
    for ep in range(n_eps):
        env.reset()
        while True:
            acts = {a: env.action_space(a).sample() for a in agents}
            _, _, costs, term, trunc, infos = env.step(acts)
            steps += 1
            if real_contacts() > 0:
                real_steps += 1
            if (infos["agent_0"].get("agent_0", {}) or {}).get("cost_contact_other"):
                key_steps += 1
            d = np.linalg.norm(data.xpos[ant_roots[0]][:2] - data.xpos[ant_roots[1]][:2])
            min_d = min(min_d, float(d))
            if any(term.values()) or any(trunc.values()):
                break
    print(f"steps: {steps} | real ant-ant contact steps: {real_steps} | "
          f"cost_contact_other steps: {key_steps} | min root-root distance: {min_d:.3f}")
except Exception as e:
    print(f"step 3 failed: {type(e).__name__}: {e}")

print("\n=== Step 4: teleport ant B next to ant A, call calculate_cost() ===")
try:
    env.reset()
    qadr = {}
    for j in range(model.njnt):
        if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and int(model.jnt_bodyid[j]) in ant_roots:
            qadr[int(model.jnt_bodyid[j])] = int(model.jnt_qposadr[j])
    print(f"free-joint qpos addresses for the two ants: {qadr}")
    a, b = ant_roots
    data.qpos[qadr[b]:qadr[b] + 3] = data.qpos[qadr[a]:qadr[a] + 3] + np.array([0.3, 0.0, 0.0])
    mujoco.mj_forward(model, data)
    cost = task.calculate_cost()
    print(f"real ant-ant contacts after teleport: {real_contacts()}")
    print(f"cost_contact_other: agent_0={cost['agent_0'].get('cost_contact_other')} "
          f"agent_1={cost['agent_1'].get('cost_contact_other')}")
except Exception as e:
    print(f"step 4 failed: {type(e).__name__}: {e}")