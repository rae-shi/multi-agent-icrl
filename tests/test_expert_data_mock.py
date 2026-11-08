"""Test of scripts/pipeline/generate_expert_data.py against the mock Safety-Gymnasium, through the REAL bundled
DummyVecEnv, and through the REAL ``load_expert_data`` function of the ICRL code (extracted from source).

Run from the repo root:  python tests/test_expert_data_mock.py
                         [--icrl algos/icrl] [--adapter envs/wrappers/safety_multigoal_pair.py]
                         [--generator scripts/pipeline/generate_expert_data.py]
"""
import argparse, ast, importlib.util, os, pickle, sys, tempfile, types
from pathlib import Path
from unittest.mock import MagicMock
import numpy as np

root = Path(__file__).resolve().parents[1]
ap = argparse.ArgumentParser()
ap.add_argument("--icrl", default=str(root / "algos" / "icrl"))
ap.add_argument("--adapter", default=str(root / "envs" / "wrappers" / "safety_multigoal_pair.py"))
ap.add_argument("--generator", default=str(root / "scripts" / "pipeline" / "generate_expert_data.py"))
args = ap.parse_args()

# ---- minimal gym shim + the real bundled vec_env files (no package __init__, no torch) ----
gym = types.ModuleType("gym"); spaces = types.ModuleType("gym.spaces")
class Space: pass
class Box(Space):
    def __init__(self, low, high, shape=None, dtype=np.float32):
        self.low = np.asarray(low, dtype); self.high = np.asarray(high, dtype)
        self.shape = self.low.shape; self.dtype = np.dtype(dtype)
class Dict(Space): pass
class Tuple(Space): pass
class Env: metadata = {}
for c in (Space, Box, Dict, Tuple): setattr(spaces, c.__name__, c)
gym.Env, gym.spaces, gym.Space = Env, spaces, Space
sys.modules["gym"], sys.modules["gym.spaces"] = gym, spaces
def fake_pkg(name, path):
    m = types.ModuleType(name); m.__path__ = [path]; sys.modules[name] = m; return m
sb3 = fake_pkg("stable_baselines3", f"{args.icrl}/stable_baselines3")
common = fake_pkg("stable_baselines3.common", f"{args.icrl}/stable_baselines3/common")
vec = fake_pkg("stable_baselines3.common.vec_env", f"{args.icrl}/stable_baselines3/common/vec_env")
sb3.common = common; common.vec_env = vec
common.logger = MagicMock(); sys.modules["stable_baselines3.common.logger"] = common.logger
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper
from stable_baselines3.common.vec_env.dummy_vec_env import DummyVecEnv

def load(path, name):
    sp = importlib.util.spec_from_file_location(name, path); m = importlib.util.module_from_spec(sp)
    sys.modules[name] = m; sp.loader.exec_module(m); return m
pair = load(args.adapter, "safety_multigoal_pair")
gen = load(args.generator, "generate_expert_data")

# ---- mock Safety-Gymnasium: obs = arange(208)+t; reward agent_0 = 1.0*t, agent_1 = 0.5*t; T-step horizon ----
class FakeSpace:
    def __init__(self, n, lo=-1.0, hi=1.0):
        self.low = np.full(n, lo, np.float32); self.high = np.full(n, hi, np.float32); self.shape = (n,)
class FakeTask: contact_other_cost = 1.0
class FakeSafety:
    def __init__(self, T=5):
        self.T, self.t, self.reset_seeds, self.task = T, 0, [], FakeTask()
    unwrapped = property(lambda self: self)
    def observation_space(self, a): return FakeSpace(208, -np.inf, np.inf)
    def action_space(self, a): return FakeSpace(8)
    def _obs(self):
        o = np.arange(208, dtype=np.float64) + self.t; return {"agent_0": o.copy(), "agent_1": o.copy()}
    def reset(self, seed=None, options=None):
        self.t = 0; self.reset_seeds.append(seed); return self._obs(), {"agent_0": {}, "agent_1": {}}
    def step(self, acts):
        self.t += 1; trunc = self.t >= self.T
        cost = {"agent_0": 0.0, "agent_1": 0.9 if self.t == 3 else 0.0}   # only agent_1 violates, at t = 3
        return (self._obs(), {"agent_0": np.array(1.0 * self.t), "agent_1": np.array(0.5 * self.t)}, cost,
                {a: False for a in cost}, {a: trunc for a in cost}, {})
    def close(self): pass

class FakeNormalize(VecEnvWrapper):
    """stands in for VecNormalizeWithCost: the policy sees obs * 0.5, get_original_obs() the raw ones."""
    def reset(self):
        self._raw = self.venv.reset(); return self._raw * 0.5
    def step_wait(self):
        obs, rew, dones, infos = self.venv.step_wait(); self._raw = obs; return obs * 0.5, rew, dones, infos
    def get_original_obs(self): return self._raw
class FakeModel:
    def __init__(self): self.seen = []
    def predict(self, obs, state=None, deterministic=True):
        self.seen.append(np.array(obs)); return np.full((obs.shape[0], 8), 0.25, np.float32), None

ok = 0
def check(cond, msg):
    global ok
    assert cond, "FAILED: " + msg
    ok += 1; print("ok  -", msg)

inner = DummyVecEnv([lambda: pair.SafetyMultiGoalPair(level=1, inner_env=FakeSafety(T=5))])
env = FakeNormalize(pair.TwoRowVecEnv(inner, env_id="SafetyAntMultiGoal1Pair-v0", log_dir=None))
model = FakeModel()
trajs = gen.collect_expert_episodes(env, model, n_episodes=3, deterministic=True, seed=5)

check(len(trajs) == 6 and [(t["episode"], t["agent"]) for t in trajs] == [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
      "3 episodes x 2 agents = 6 trajectories, ordered episode-major, agent alternating")
t = trajs[0]
check(t["original_observations"].shape == (5, 210) and t["actions"].shape == (5, 8) and t["reward"].shape == (5,), "shapes (T,210) / (T,8) / (T,)")
check(np.allclose(t["original_observations"][0, :208], np.arange(208)) and np.allclose(t["original_observations"][2, :208], np.arange(208) + 2),
      "stored observations are the ORIGINAL (unnormalized) ones")
check(np.allclose(model.seen[0][0, :208], np.arange(208) * 0.5), "the policy was fed the NORMALIZED observations")
check(list(trajs[0]["original_observations"][0, 208:210]) == [1, 0] and list(trajs[1]["original_observations"][0, 208:210]) == [0, 1],
      "agent 0's trajectory carries id [1,0], agent 1's carries [0,1]")
check(np.allclose(trajs[2]["original_observations"][0, :208], np.arange(208)), "episode 1 starts from a fresh reset, not from the terminal observation")
check(np.allclose(t["actions"], 0.25), "actions stored as the policy chose them")
check(abs(trajs[0]["reward_sum"] - 15.0) < 1e-6 and abs(trajs[1]["reward_sum"] - 7.5) < 1e-6, "per-agent returns (15.0 and 7.5)")
check(trajs[0]["lag_cost"].sum() == 0 and list(trajs[1]["lag_cost"]) == [0, 0, 1, 0, 0], "violations recorded per agent, independently")
check(inner.envs[0]._env.reset_seeds[:2] == [None, 5], "seed passed to the first reset only")
summary = gen.summarize(trajs)
check(len(summary) == 2 and "agent_0" in summary[0] and "episodes with a violation 0/3" in summary[0] and "episodes with a violation 3/3" in summary[1],
      "summary: agent 0 never violates, agent 1 violates in 3/3 episodes")

# ---- the files must load through the REAL load_expert_data of the ICRL code ----
out = tempfile.mkdtemp()
gen.save_trajectories(trajs, out)
check(sorted(os.listdir(out)) == ["ep0000_a0.pkl", "ep0000_a1.pkl", "ep0001_a0.pkl", "ep0001_a1.pkl", "ep0002_a0.pkl", "ep0002_a1.pkl"], "file names")
src = open(f"{args.icrl}/utils/data_utils.py").read()
fn = next(n for n in ast.parse(src).body if isinstance(n, ast.FunctionDef) and n.name == "load_expert_data")
ns = {"np": np, "os": os, "pickle": pickle, "pickle5": None}
exec(compile(ast.Module(body=[fn], type_ignores=[]), "load_expert_data", "exec"), ns)
(eo, ea, er), mean_reward = ns["load_expert_data"](expert_path=out, num_rollouts=None, add_next_step=False, log_file=None)
check(len(eo) == 6 and all(x.shape == (5, 210) for x in eo) and all(x.shape == (5, 8) for x in ea), "real load_expert_data: 6 rollouts, (5,210) observations, (5,8) actions")
check(abs(mean_reward - 11.25) < 1e-6, "real load_expert_data: mean reward over trajectories = 11.25")
(eo4, _, _), _ = ns["load_expert_data"](expert_path=out, num_rollouts=4, add_next_step=False, log_file=None)
check(len(eo4) == 4 and list(eo4[0][0, 208:210]) == [1, 0] and list(eo4[1][0, 208:210]) == [0, 1] and list(eo4[2][0, 208:210]) == [1, 0],
      "expert_rollouts: 4 loads the first 2 episodes, both agents (alternating ids)")
print(f"\nALL {ok} CHECKS PASSED")
