"""One-off patch: optional fall penalty on the reward, training only. Run from the repo root.

    python scripts/apply_fall_penalty_patch.py            # patch in place
    git diff                                              # review

Adds `fall_penalty` (default 0.0 = unchanged behaviour):
  - envs/wrappers/safety_multigoal_pair.py : penalty on the reward of both rows when the episode ends by
    termination (not the 1000-step limit); original reward kept as rows[i]["reward_raw"]; info["fell"];
    the monitor files keep recording the ORIGINAL reward.
  - algos/icrl/common/cns_env.py : training env only (the evaluation env is not touched -> original reward).
  - algos/icrl/interface/train_policy.py : reads config['env']['fall_penalty'] (default 0.0).
Every replacement must match exactly once or nothing is written.
"""
import re
import sys

PAIR = "envs/wrappers/safety_multigoal_pair.py"
CNS = "algos/icrl/common/cns_env.py"
TRAIN = "algos/icrl/interface/train_policy.py"


def sub_once(s, old, new, what):
    n = s.count(old)
    if n != 1:
        sys.exit("PATCH FAILED ({0}): expected 1 match, found {1}. Nothing written.".format(what, n))
    return s.replace(old, new)


def patch_pair(s):
    s = sub_once(s, "    def __init__(self, level=1, inner_env=None):",
                 "    def __init__(self, level=1, inner_env=None, fall_penalty=0.0):", "adapter __init__ signature")
    s = sub_once(s, "        self.level = level\n",
                 "        self.level = level\n        self._fall_penalty = float(fall_penalty)   # reward penalty when the episode ends by a fall\n",
                 "adapter self.level")
    old = '''        rows = []
        for a in self.agents:
            c = float(costs[a])
            rows.append({"reward": float(rewards[a]), "cost": c, "lag_cost": int(c > 0)})
        term = any(bool(terminated[a]) for a in self.agents)
        trunc = any(bool(truncated[a]) for a in self.agents)
        info = {"rows": rows,
'''
    new = '''        term = any(bool(terminated[a]) for a in self.agents)
        trunc = any(bool(truncated[a]) for a in self.agents)
        fell = bool(term and not trunc)       # ended by termination (a fall), not by the 1000-step limit
        rows = []
        for a in self.agents:
            c = float(costs[a])
            raw = float(rewards[a])
            rows.append({"reward": raw - (self._fall_penalty if fell else 0.0), "reward_raw": raw,
                         "cost": c, "lag_cost": int(c > 0)})
        info = {"rows": rows,
                "fell": fell,
'''
    s = sub_once(s, old, new, "adapter step")
    s = sub_once(s, 'self._monitor_step(row, ri["reward"], ri["lag_cost"])',
                 'self._monitor_step(row, ri.get("reward_raw", ri["reward"]), ri["lag_cost"])', "monitor reward")
    s = sub_once(s, "env_factory=None, seed_fn=None, start_method=None):",
                 "env_factory=None, seed_fn=None, start_method=None, fall_penalty=0.0):", "make_pair_vec_env signature")
    s = sub_once(s, "else SafetyMultiGoalPair(level=level)",
                 "else SafetyMultiGoalPair(level=level, fall_penalty=fall_penalty)", "make_pair_vec_env factory")
    return s


def patch_cns(s):
    old = """        env = make_pair_vec_env(env_id, num_threads=num_threads, base_seed=base_seed, log_dir=save_dir,
                                parallel=True)"""
    new = """        env = make_pair_vec_env(env_id, num_threads=num_threads, base_seed=base_seed, log_dir=save_dir,
                                parallel=True, fall_penalty=kwargs.get('fall_penalty', 0.0))"""
    return sub_once(s, old, new, "cns_env training call")


def patch_train(s):
    start = s.find("make_train_env(")
    if start < 0:
        sys.exit("PATCH FAILED (train_policy): make_train_env( not found. Nothing written.")
    m = re.compile(r"\n([ \t]*)cost_gamma=config\['env'\]\['cost_gamma'\],").search(s, start)
    if not m:
        sys.exit("PATCH FAILED (train_policy): cost_gamma line not found. Nothing written.")
    ins = "\n{0}fall_penalty=config['env'].get('fall_penalty', 0.0),".format(m.group(1))
    return s[:m.end()] + ins + s[m.end():]


def main():
    plan = [(PAIR, patch_pair), (CNS, patch_cns), (TRAIN, patch_train)]
    new = {}
    for path, fn in plan:
        s = open(path).read()
        if "fall_penalty" in s:
            sys.exit("{0} already contains fall_penalty; patch seems applied. Nothing written.".format(path))
        new[path] = fn(s)
    for path, s in new.items():
        open(path, "w").write(s)
        print("patched", path)


if __name__ == "__main__":
    main()