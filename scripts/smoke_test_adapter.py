#!/usr/bin/env python3
"""Smoke test: the MultiGoal joint-action adapter and the patched ICRL code path.

Run it as a job, from the repo root, with the repo root and algos/icrl on PYTHONPATH:

    export PYTHONPATH="$PWD:$PWD/algos/icrl:$PYTHONPATH"
    srun --account=def-s2ganapa --time=00:15:00 --mem=4G python scripts/smoke_test_adapter.py

Optional argument: the env id (default SafetyAntMultiGoal1Joint-v0).

Part 1 checks the raw environment (spaces, 4-tuple step, cost keys, speed).
Part 2 builds the env with the benchmark's own factory (external-signal wrapper + CNSMonitor)
and checks that the monitor's `constraint` column agrees with the cost the environment gave.
"""
import csv
import os
import sys
import tempfile
import time

os.environ.setdefault("MUJOCO_GL", "egl")
os.environ.setdefault("PYOPENGL_PLATFORM", "egl")

import numpy as np
import gym

import mujuco_environment.custom_envs  # noqa: F401  (registers the ids; patched)
from common.cns_env import make_env
from utils.env_utils import is_mujoco

ENV_ID = sys.argv[1] if len(sys.argv) > 1 else "SafetyAntMultiGoal1Joint-v0"
RAW_STEPS = 600
MONITOR_EPISODES = 3

failures = []


def check(cond, msg):
    print(("ok   - " if cond else "FAIL - ") + msg, flush=True)
    if not cond:
        failures.append(msg)


# ------------------------------------------------------------------ part 1
print("== 1. raw environment: {0} ==".format(ENV_ID), flush=True)
check(is_mujoco(ENV_ID), "is_mujoco accepts the env id (patch 0001 applied)")
env = gym.make(ENV_ID)
print("observation_space:", env.observation_space, flush=True)
print("action_space     :", env.action_space, flush=True)
check(env.observation_space.shape == (208,), "observation is 208-d")
check(env.action_space.shape == (16,), "joint action is 16-d (2 x 8)")

obs = env.reset()
check(isinstance(obs, np.ndarray) and obs.shape == (208,), "reset returns one 208-d observation")

steps = cost_steps = episodes = 0
ep_len = 0
ep_lens = []
bad_keys = False
t0 = time.time()
for _ in range(RAW_STEPS):
    obs, reward, done, info = env.step(env.action_space.sample())
    steps += 1
    ep_len += 1
    if not all(k in info for k in ("cost", "lag_cost", "cost_agent_0", "cost_agent_1")):
        bad_keys = True
    if info["lag_cost"]:
        cost_steps += 1
    if done:
        episodes += 1
        ep_lens.append(ep_len)
        ep_len = 0
        obs = env.reset()
elapsed = time.time() - t0
check(not bad_keys, "info always has cost, lag_cost, cost_agent_0, cost_agent_1")
print("   {0} steps in {1:.1f}s = {2:.0f} steps/s; episodes finished: {3} (lengths {4}); "
      "steps with cost > 0: {5} ({6:.1f}%)".format(steps, elapsed, steps / elapsed, episodes,
                                                  ep_lens, cost_steps, 100.0 * cost_steps / steps),
      flush=True)
env.close()

# ------------------------------------------------------------------ part 2
print("\n== 2. through the benchmark's own env factory (wrappers + CNSMonitor) ==", flush=True)
log_dir = tempfile.mkdtemp(prefix="smoke_monitor_")
menv = make_env(env_id=ENV_ID, env_configs={}, rank=0, log_dir=log_dir, group="PPO-Lag")()
obs = menv.reset()
episode_flags = []
ep_any = 0
t0 = time.time()
while len(episode_flags) < MONITOR_EPISODES:
    obs, reward, done, info = menv.step(menv.action_space.sample())
    if "lag_cost" not in info:
        check(False, "info lost lag_cost after the wrappers")
        break
    ep_any = max(ep_any, int(info["lag_cost"]))
    if done:
        episode_flags.append(ep_any)
        ep_any = 0
        obs = menv.reset()
menv.close()
print("   {0} episodes in {1:.1f}s; per-episode 'any violation' from info: {2}".format(
    len(episode_flags), time.time() - t0, episode_flags), flush=True)

csv_files = [f for f in os.listdir(log_dir) if f.endswith("monitor.csv")]
check(len(csv_files) == 1, "the monitor wrote one monitor.csv in " + log_dir)
if csv_files:
    with open(os.path.join(log_dir, csv_files[0])) as fh:
        lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]
    print("   monitor.csv (first rows):", flush=True)
    for l in lines[:5]:
        print("     " + l, flush=True)
    if lines and lines[0].startswith("reward"):
        header, data = lines[0].split(","), lines[1:]
    else:
        header, data = ["reward", "reward_nc", "len", "time", "constraint"], lines
    rows = list(csv.reader(data))
    check(len(rows) == len(episode_flags), "one monitor row per episode ({0})".format(len(rows)))
    if "constraint" in header and len(rows) == len(episode_flags):
        col = header.index("constraint")
        flags = [int(float(r[col])) for r in rows]
        check(flags == episode_flags,
              "monitor 'constraint' column {0} matches the environment's own cost {1}".format(
                  flags, episode_flags))
    else:
        check(False, "could not find the 'constraint' column in monitor.csv")

print("\n" + ("SMOKE TEST PASSED" if not failures else "SMOKE TEST FAILED: {0}".format(failures)))
sys.exit(1 if failures else 0)
