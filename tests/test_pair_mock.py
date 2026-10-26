"""Unit test of the two-row adapter (envs/wrappers/safety_multigoal_pair.py) against a MOCK of the
Safety-Gymnasium API, run through the REAL bundled DummyVecEnv / SubprocVecEnv and through the
patched evaluation / sampler functions of the ICRL code.

Needs only numpy + cloudpickle: `gym` is shimmed, torch/matplotlib are stubbed. It checks our own
logic; it cannot detect a difference between the mock and the real environment
(scripts/smoke_test_pair.py is the test against the real one).

Run from the repo root (patches 0001-0006 must be applied to algos/icrl):
    python tests/test_pair_mock.py                         # uses ./algos/icrl and ./envs
    python tests/test_pair_mock.py --icrl /path/to/icrl --adapter /path/to/safety_multigoal_pair.py
"""
import argparse, csv, importlib.util, os, sys, tempfile, types
from collections import OrderedDict
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

ap = argparse.ArgumentParser()
root = Path(__file__).resolve().parents[1]
ap.add_argument("--icrl", default=str(root / "algos" / "icrl"))
ap.add_argument("--adapter", default=str(root / "envs" / "wrappers" / "safety_multigoal_pair.py"))
args = ap.parse_args()

# ---------------------------------------------------------------- shim: gym ----------------
gym = types.ModuleType("gym"); spaces = types.ModuleType("gym.spaces")
class Space: pass
class Box(Space):
    def __init__(self, low, high, shape=None, dtype=np.float32):
        self.low = np.asarray(low, dtype); self.high = np.asarray(high, dtype)
        self.shape = self.low.shape; self.dtype = np.dtype(dtype)
class Discrete(Space): pass
class MultiDiscrete(Space): pass
class MultiBinary(Space): pass
class Dict(Space):
    def __init__(self, sp): self.spaces = sp
class Tuple(Space):
    def __init__(self, sp): self.spaces = sp
class Env:
    metadata = {}
for c in (Space, Box, Discrete, MultiDiscrete, MultiBinary, Dict, Tuple): setattr(spaces, c.__name__, c)
gym.Env = Env; gym.spaces = spaces; gym.Space = Space
sys.modules["gym"] = gym; sys.modules["gym.spaces"] = spaces

# ------------------------- real bundled vec_env files, loaded without the package __init__ ---
def fake_pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m; return m
icrl = args.icrl
sb3 = fake_pkg("stable_baselines3", f"{icrl}/stable_baselines3")
common = fake_pkg("stable_baselines3.common", f"{icrl}/stable_baselines3/common")
vec = fake_pkg("stable_baselines3.common.vec_env", f"{icrl}/stable_baselines3/common/vec_env")
sb3.common = common; common.vec_env = vec
common.logger = MagicMock(); sys.modules["stable_baselines3.common.logger"] = common.logger
from stable_baselines3.common.vec_env.base_vec_env import VecEnv, VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv
from stable_baselines3.common.vec_env.subproc_vec_env import SubprocVecEnv
vec.VecEnv, vec.VecEnvWrapper, vec.DummyVecEnv, vec.SubprocVecEnv = VecEnv, VecEnvWrapper, DummyVecEnv, SubprocVecEnv
vec.sync_envs_normalization = MagicMock()

spec = importlib.util.spec_from_file_location("safety_multigoal_pair", args.adapter)
pair = importlib.util.module_from_spec(spec); sys.modules["safety_multigoal_pair"] = pair
spec.loader.exec_module(pair)

# ---------------------------------------------------------------- mock Safety-Gymnasium ----
class FakeSpace:
    def __init__(self, n, lo=-1.0, hi=1.0):
        self.low = np.full(n, lo, np.float32); self.high = np.full(n, hi, np.float32); self.shape = (n,)
class FakeTask:
    contact_other_cost = 1.0
class FakeSafety:
    """obs = arange(208)+t (shared); reward agent_0 = 1.0*t, agent_1 = 0.5*t; truncated at T.
    cost_fn(episode, t, agent) -> cost.  reset() rebuilds the task (cost reverts to 1.0)."""
    def __init__(self, T=5, cost_fn=None, terminate_at=None, differing_obs=False, bad_action_space=False):
        self.T, self.t, self.ep = T, 0, -1
        self.cost_fn = cost_fn or (lambda ep, t, a: 0.0)
        self.terminate_at, self.differing, self.bad_as = terminate_at, differing_obs, bad_action_space
        self.seen_actions, self.reset_seeds = [], []
        self.task = FakeTask(); self.closed = False
    unwrapped = property(lambda self: self)
    def observation_space(self, a): return FakeSpace(208, -np.inf, np.inf)
    def action_space(self, a): return FakeSpace(8, -2.0 if (self.bad_as and a == "agent_1") else -1.0)
    def _obs(self):
        o = np.arange(208, dtype=np.float64) + self.t
        d = {"agent_0": o.copy(), "agent_1": o.copy()}
        if self.differing: d["agent_1"][0] += 1
        return d
    def reset(self, seed=None, options=None):
        self.t = 0; self.ep += 1; self.reset_seeds.append(seed)
        self.task = FakeTask()                      # a rebuilt task: contact cost reverts to 1.0
        return self._obs(), {"agent_0": {}, "agent_1": {}}
    def step(self, acts):
        self.seen_actions.append({k: np.array(v) for k, v in acts.items()}); self.t += 1
        term = self.terminate_at is not None and self.t >= self.terminate_at
        trunc = self.t >= self.T and not term
        rew = {"agent_0": np.array(1.0 * self.t), "agent_1": np.array(0.5 * self.t)}   # 0-d arrays, like the real env
        cost = {a: float(self.cost_fn(self.ep, self.t, a)) for a in ("agent_0", "agent_1")}
        return (self._obs(), rew, cost, {a: term for a in cost}, {a: trunc for a in cost}, {})
    def render(self): return "rendered"
    def close(self): self.closed = True

ok = 0
def check(cond, msg):
    global ok
    assert cond, "FAILED: " + msg
    ok += 1; print("ok  -", msg)

def agent1_violates_at_t3_in_first_episode(ep, t, a):
    # episode 0 is the constructor's own reset; the first episode a test runs is episode 1
    return 0.7 if (ep == 1 and t == 3 and a == "agent_1") else 0.0

# ================================================ 1. SafetyMultiGoalPair (one simulation) =====
fake = FakeSafety(T=5, cost_fn=agent1_violates_at_t3_in_first_episode)
check(fake.task.contact_other_cost == 1.0, "mock starts with the agent-contact cost ON (1.0)")
env = pair.SafetyMultiGoalPair(level=1, inner_env=fake)
check(fake.task.contact_other_cost == 0.0 and env.contact_cost_disabled, "contact_other_cost switched off at construction")
check(env.agents == ["agent_0", "agent_1"], "agent ids taken from the reset observation")
check(env.observation_space.shape == (420,) and env.observation_space.dtype == np.float32, "flat observation = 2 rows x 210")
check(np.all(env.observation_space.low[208:210] == 0) and np.all(env.observation_space.high[208:210] == 1)
      and np.all(env.observation_space.high[418:420] == 1), "one-hot dims bounded in [0,1]")
check(env.action_space.shape == (16,), "joint action space is 16-d (2 x 8)")

flat = env.reset()
base = np.arange(208, dtype=np.float32)
check(flat.shape == (420,) and flat.dtype == np.float32, "reset returns one float32 vector of 420")
check(np.array_equal(flat[:208], base) and np.array_equal(flat[210:418], base), "both rows carry the shared 208-d observation")
check(list(flat[208:210]) == [1, 0] and list(flat[418:420]) == [0, 1], "one-hot id: row 0 = [1,0], row 1 = [0,1]")
check(fake.task.contact_other_cost == 0.0, "contact cost stays off after reset() even though the mock rebuilt its task")

a = np.concatenate([np.full(8, 0.1), np.full(8, -0.2)])
obs, reward, done, info = env.step(a)
check(np.allclose(fake.seen_actions[-1]["agent_0"], 0.1) and np.allclose(fake.seen_actions[-1]["agent_1"], -0.2),
      "joint action split: first 8 -> agent_0, last 8 -> agent_1")
check([r["reward"] for r in info["rows"]] == [1.0, 0.5], "per-row rewards are kept separate (0-d arrays handled)")
check(all(r["cost"] == 0.0 and r["lag_cost"] == 0 for r in info["rows"]) and done is False, "no cost, not done at t=1")
env.step(a)
_, _, _, info = env.step(a)    # t = 3: only agent_1 violates
check(info["rows"][0]["lag_cost"] == 0 and info["rows"][1]["lag_cost"] == 1, "INDEPENDENT constraints: agent_1 violation does not flag agent_0")
check(info["rows"][1]["cost"] == 0.7 and info["rows"][0]["cost"] == 0.0, "each row keeps its own cost value")
check(info["lag_cost"] == 1 and abs(info["cost"] - 0.7) < 1e-9, "joint summary fields still available")
env.step(a); _, _, done, info = env.step(a)
check(done is True and info["TimeLimit.truncated"] is True, "truncation at the horizon -> done, flagged as time-limit")
f2 = pair.SafetyMultiGoalPair(level=1, inner_env=FakeSafety(T=5, terminate_at=2)); f2.reset(); f2.step(a)
_, _, done, info = f2.step(a)
check(done is True and info["TimeLimit.truncated"] is False, "true termination is not flagged as truncation")
env.seed(123); env.reset(); env.reset()
check(fake.reset_seeds[-2:] == [123, None], "seed is passed to the next reset only")
for bad, exc, msg in [(lambda: env.step(np.zeros(8)), ValueError, "wrong action length raises ValueError"),
                      (lambda: pair.SafetyMultiGoalPair(1, FakeSafety(differing_obs=True)), RuntimeError, "different per-agent observations raise"),
                      (lambda: pair.SafetyMultiGoalPair(1, FakeSafety(bad_action_space=True)), ValueError, "different per-agent action spaces raise (shared policy invalid)")]:
    try: bad(); check(False, msg)
    except exc: check(True, msg)
check(pair.is_pair_env_id("SafetyAntMultiGoal1Pair-v0") and not pair.is_pair_env_id("SafetyAntMultiGoal1Joint-v0")
      and pair.level_from_env_id("SafetyAntMultiGoal2Pair-v0") == 2, "env id helpers")

# ===================================== 2. TwoRowVecEnv over the REAL DummyVecEnv / SubprocVecEnv ==
def factory():
    return pair.SafetyMultiGoalPair(level=1, inner_env=FakeSafety(T=5, cost_fn=agent1_violates_at_t3_in_first_episode))

def build(parallel, log_dir):
    return pair.make_pair_vec_env("SafetyAntMultiGoal1Pair-v0", num_threads=2, base_seed=7, log_dir=log_dir,
                                  parallel=parallel, env_factory=factory, seed_fn=lambda s: None,
                                  start_method="fork" if parallel else None)

def run_episode_pair(venv):
    """reset, then step 10 times (two 5-step episodes) with row r getting action value 0.1*(r+1)."""
    obs0 = venv.reset(); out = []
    for t in range(10):
        venv.step_async(np.repeat(np.arange(1, venv.num_envs + 1)[:, None] * 0.1, 8, axis=1).astype(np.float32))
        out.append(venv.step_wait())
    return obs0, out

results = {}
for mode, parallel in [("dummy", False), ("subproc", True)]:
    d = tempfile.mkdtemp() + os.sep
    venv = build(parallel, d)
    check(venv.num_envs == 4 and venv.n_sims == 2, f"[{mode}] 2 simulations -> 4 rows")
    check(venv.observation_space.shape == (210,) and venv.action_space.shape == (8,), f"[{mode}] per-row spaces: 210-d obs, 8-d action")
    obs0, out = run_episode_pair(venv)
    check(obs0.shape == (4, 210) and [list(o[208:210]) for o in obs0] == [[1, 0], [0, 1], [1, 0], [0, 1]], f"[{mode}] rows 2s / 2s+1 are agent 0 / agent 1")
    obs1, rew1, done1, info1 = out[0]
    check(np.allclose(rew1, [1.0, 0.5, 1.0, 0.5]), f"[{mode}] per-row rewards")
    check(not done1.any() and len(info1) == 4, f"[{mode}] not done, one info per row")
    _, _, _, info3 = out[2]    # t = 3 of the first episode: agent_1 violates in every simulation, agent_0 never
    check([i["lag_cost"] for i in info3] == [0, 1, 0, 1], f"[{mode}] violation appears on the agent-1 rows only (rows 1 and 3)")
    check(abs(info3[1]["cost"] - 0.7) < 1e-9 and info3[0]["cost"] == 0.0, f"[{mode}] per-row cost values")
    obs5, rew5, done5, info5 = out[4]   # t = 5: horizon
    check(done5.all() and all("terminal_observation" in i for i in info5), f"[{mode}] both rows end together with terminal observations")
    check([list(i["terminal_observation"][208:210]) for i in info5] == [[1, 0], [0, 1], [1, 0], [0, 1]], f"[{mode}] terminal obs split per row, right one-hot")
    check(np.allclose(info5[0]["terminal_observation"][:208], np.arange(208) + 5) and np.allclose(obs5[0][:208], np.arange(208)),
          f"[{mode}] terminal obs is the last obs; returned obs is the auto-reset one")
    check(all(i["episode"]["len"] == 5 for i in info5), f"[{mode}] episode info present, len 5")
    venv.close()
    rows = {}
    for r in range(4):
        with open(os.path.join(d, f"r{r}_monitor.csv")) as fh:
            lines = fh.read().splitlines()
        check(lines[0].startswith("#") and lines[1] == "reward,reward_nc,len,time,constraint", f"[{mode}] r{r}_monitor.csv header format")
        rows[r] = list(csv.DictReader(lines[1:]))
    check(all(len(rows[r]) == 2 for r in range(4)), f"[{mode}] two episodes per row monitor")
    e0 = {r: rows[r][0] for r in range(4)}
    check(e0[0]["reward"] == "15.0" and e0[0]["reward_nc"] == "15.0" and e0[0]["constraint"] == "0", f"[{mode}] agent 0: no violation -> reward_nc = reward")
    check(e0[1]["reward"] == "7.5" and e0[1]["reward_nc"] == "1.5" and e0[1]["constraint"] == "1",
          f"[{mode}] agent 1: reward 7.5, reward_nc 1.5 (violating step's reward excluded), constraint 1")
    check(e0[2]["constraint"] == "0" and e0[3]["constraint"] == "1" and e0[3]["reward_nc"] == "1.5",
          f"[{mode}] second simulation behaves the same: agent 0 clean, agent 1 violates")
    check(all(rows[r][1]["constraint"] == "0" for r in range(4)), f"[{mode}] violation flag cleared for the next episode")
    results[mode] = (obs0, out, rows)

o_d, out_d, rows_d = results["dummy"]; o_s, out_s, rows_s = results["subproc"]
check(np.array_equal(o_d, o_s) and all(np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1]) and np.array_equal(a[2], b[2])
      and [i["lag_cost"] for i in a[3]] == [i["lag_cost"] for i in b[3]] for a, b in zip(out_d, out_s)),
      "DummyVecEnv and SubprocVecEnv give identical obs / rewards / dones / violation flags")
check(all([{k: v for k, v in x.items() if k != "time"} for x in rows_d[r]] == [{k: v for k, v in x.items() if k != "time"} for x in rows_s[r]] for r in range(4)),
      "monitor files identical between the two modes")

# =============== 3. the patched evaluation / sampler functions of the ICRL code, on 2 rows ========
for name in ["torch", "matplotlib", "matplotlib.pyplot", "stable_baselines3.common.callbacks",
             "stable_baselines3.common.evaluation", "planner", "planner.cross_entropy_method",
             "planner.cross_entropy_method.cem", "utils", "utils.env_utils"]:
    sys.modules.setdefault(name, MagicMock())
def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp); sys.modules[name] = m
    sp.loader.exec_module(m); return m
ev = load(f"{icrl}/common/cns_evaluation.py", "cns_evaluation")
sm = load(f"{icrl}/common/cns_sampler.py", "cns_sampler")

class WithOriginal(pair.TwoRowVecEnv):          # stands in for VecNormalize.get_original_obs
    def step_wait(self):
        out = super().step_wait(); self._last = out[0]; return out
    def reset(self):
        self._last = super().reset(); return self._last
    def get_original_obs(self): return self._last
class ZeroModel:
    def predict(self, obs, state=None, deterministic=True): return np.zeros((obs.shape[0], 8), np.float32), None
class RandModel:
    def predict(self, obs, state=None, deterministic=False): return np.random.uniform(-1, 1, (obs.shape[0], 8)).astype(np.float32), None

def one_sim_env(cls=pair.TwoRowVecEnv):
    inner = DummyVecEnv([factory]); return cls(inner, env_id="SafetyAntMultiGoal1Pair-v0", log_dir=None)

e = one_sim_env()
mean_r, std_r, mean_nc, std_nc, rec, costs = ev.evaluate_icrl_policy(ZeroModel(), e, record_info_names=[], n_eval_episodes=2)
check(True, "evaluation runs on 2 rows without error")
check(abs(mean_r - 11.25) < 1e-9, "evaluation mean return = mean over both agents' rows (11.25)")
check(abs(mean_nc - ((15 + 1.5 + 15 + 7.5) / 4)) < 1e-9, "evaluation reward_nc: only agent 1's episode-0 return is cut at its violation")
try:
    ev.evaluate_icrl_policy(ZeroModel(), pair.make_pair_vec_env("SafetyAntMultiGoal1Pair-v0", 2, parallel=False, env_factory=factory, seed_fn=lambda s: None), [], 1)
    check(False, "evaluation must still refuse more than one simulation")
except AssertionError: check(True, "evaluation still refuses more than one simulation (4 rows)")

for by_game in (True, False):
    e = one_sim_env(WithOriginal)
    s = sm.ConstrainedRLSampler(rollouts=2, store_by_game=by_game, cost_info_str="cost", env=e, sample_multi_env=False, env_id="x")
    o_orig, o, acs, rs, sums, lens = s.sample_from_agent(RandModel(), e)
    if by_game:
        check(len(o) == 4 and all(x.shape == (5, 210) for x in o), "sampler: 2 episodes x 2 rows -> 4 trajectories of shape (5, 210)")
        check(all(x.shape == (5, 8) for x in acs) and all(x.shape == (5,) for x in rs), "sampler: actions (5,8), rewards (5,)")
        check([list(x[0, 208:210]) for x in o] == [[1, 0], [0, 1], [1, 0], [0, 1]] and all((x[:, 208:210] == x[0, 208:210]).all() for x in o),
              "sampler: trajectories alternate agent 0 / agent 1; the id is constant within a trajectory")
        check([float(x[0]) for x in sums] == [15.0, 7.5, 15.0, 7.5] and lens == [5, 5, 5, 5], "sampler: per-trajectory return and length")
    else:
        check(len(o) == 10 and all(x.shape == (2, 210) for x in o), "sampler (not by game): 10 steps, each (2 rows, 210)")
print(f"\nALL {ok} CHECKS PASSED")
