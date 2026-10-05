#!/usr/bin/env python3
"""
test_ant_collision_masks.py

Follow-up to test_contact_diagnose.py to find out can ant 0 and ant 1 physically
collide at all? (test_contact_diagnose.py showed zero ant-ant contacts even with torsos teleported 0.3 apart.)

  1. contype/conaffinity of the two ants' geoms, and how many (ant0 x ant1) geom pairs
     MuJoCo's rule allows: (ct1 & ca2) or (ct2 & ca1)
  2. teleport ant 1 near ant 0 at several offsets, then list every contact that involves
     either ant, with partner names, so we see what they DO touch.

Ants are identified from the model (two root bodies with most geoms), as in v2.
Untested: attribute names come from reading source. Steps are wrapped in try/except.
"""
import sys
from collections import defaultdict
import numpy as np
import mujoco
import safety_gymnasium

env_id = sys.argv[1] if len(sys.argv) > 1 else "SafetyAntMultiGoal2-v0"
env = safety_gymnasium.make(env_id)
env.reset()
task = env.unwrapped.task
model, data = task.model, task.data


def root_of_geom(g):
    return int(model.body_rootid[int(model.geom_bodyid[int(g)])])


groups = defaultdict(list)
for g in range(model.ngeom):
    r = root_of_geom(g)
    if r != 0:
        groups[r].append(g)
ranked = sorted(groups.items(), key=lambda kv: -len(kv[1]))
A, B = ranked[0][0], ranked[1][0]
print(f"ants: root {A} '{model.body(A).name}' and root {B} '{model.body(B).name}'")

print("\n=== 1. collision masks ===")
try:
    for root in (A, B):
        sample = groups[root][:3]
        print(f"root {root}: " + ", ".join(
            f"{model.geom(g).name}(ct={int(model.geom_contype[g])},ca={int(model.geom_conaffinity[g])})"
            for g in sample))
    ok = 0
    for ga in groups[A]:
        for gb in groups[B]:
            if (int(model.geom_contype[ga]) & int(model.geom_conaffinity[gb])) or \
               (int(model.geom_contype[gb]) & int(model.geom_conaffinity[ga])):
                ok += 1
    print(f"ant0 x ant1 geom pairs allowed to collide: {ok} / {len(groups[A]) * len(groups[B])}")
    print(f"model.npair={model.npair} nexclude={model.nexclude} "
          f"contact disabled flag: {bool(model.opt.disableflags & mujoco.mjtDisableBit.mjDSBL_CONTACT)}")
except Exception as e:
    print(f"step 1 failed: {type(e).__name__}: {e}")

print("\n=== 2. teleport ant B near ant A, list contacts involving either ant ===")
try:
    qadr = {int(model.jnt_bodyid[j]): int(model.jnt_qposadr[j])
            for j in range(model.njnt)
            if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE and int(model.jnt_bodyid[j]) in (A, B)}
    for dx in (0.6, 0.3, 0.0):
        env.reset()
        data.qpos[qadr[B]:qadr[B] + 3] = data.qpos[qadr[A]:qadr[A] + 3] + np.array([dx, 0.0, 0.0])
        mujoco.mj_forward(model, data)
        print(f"-- offset {dx}: root xy A={np.round(data.xpos[A][:2], 2)} B={np.round(data.xpos[B][:2], 2)}, "
              f"ncon={data.ncon}")
        shown = 0
        for c in data.contact[: data.ncon]:
            ra, rb = root_of_geom(c.geom1), root_of_geom(c.geom2)
            if A in (ra, rb) or B in (ra, rb):
                tag = "ANT-ANT" if {ra, rb} == {A, B} else "ant-other"
                print(f"   {tag}: {model.geom(c.geom1).name} <-> {model.geom(c.geom2).name} dist={c.dist:.3f}")
                shown += 1
                if shown >= 8:
                    break
        if shown == 0:
            print("   no contacts involving either ant")
except Exception as e:
    print(f"step 2 failed: {type(e).__name__}: {e}")