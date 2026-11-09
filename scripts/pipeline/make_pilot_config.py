"""Derive a pilot config from the main Pair PPO-Lag config.

Changes only: task name, PPO.budget, running.n_iters. Everything else is copied.
YAML comments are not preserved (header line says where the file came from).
Fails loudly if an expected key is missing.
"""
import argparse
import yaml

p = argparse.ArgumentParser()
p.add_argument("--src", required=True)
p.add_argument("--dst", required=True)
p.add_argument("--task", required=True)
p.add_argument("--budget", type=float, required=True)
p.add_argument("--n-iters", type=int, required=True)
a = p.parse_args()

cfg = yaml.safe_load(open(a.src))
for path in (("task",), ("PPO", "budget"), ("running", "n_iters")):
    node = cfg
    for k in path:
        if not isinstance(node, dict) or k not in node:
            raise SystemExit(f"missing key {'.'.join(path)}; top-level keys: {list(cfg)}")
        node = node[k]

old = (cfg["task"], cfg["PPO"]["budget"], cfg["running"]["n_iters"])
cfg["task"] = a.task
cfg["PPO"]["budget"] = a.budget
cfg["running"]["n_iters"] = a.n_iters

with open(a.dst, "w") as f:
    f.write(f"# Derived from {a.src} by scripts/pipeline/make_pilot_config.py (comments not preserved)\n")
    yaml.safe_dump(cfg, f, sort_keys=False)
print(f"task {old[0]} -> {a.task}; budget {old[1]} -> {a.budget}; n_iters {old[2]} -> {a.n_iters}")
print("forward_timesteps (rows/iteration, rounded up to whole rollouts):", cfg["PPO"]["forward_timesteps"])