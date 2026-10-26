"""Unit test of envs/wrappers/safety_multigoal_joint.py against a MOCK of the Safety-Gymnasium API.

Needs only numpy: `gym` and the Safety-Gymnasium environment are faked, so it runs anywhere.
It checks the adapter's own logic (action split, summed reward/cost, lag_cost, done/truncation,
seeding, error handling). It cannot detect a difference between the mock and the real
environment; scripts/smoke_test_adapter.py is the test against the real one.

Run from the repo root:   python tests/test_adapter_mock.py
"""
import sys, types, importlib.util
import numpy as np

# ---- minimal fake `gym` (Env + spaces.Box), enough to import the adapter ----
gym = types.ModuleType("gym"); spaces = types.ModuleType("gym.spaces")
class Box:
    def __init__(self, low, high, dtype=np.float32):
        self.low = np.asarray(low, dtype); self.high = np.asarray(high, dtype)
        self.shape = self.low.shape; self.dtype = dtype
class Env: pass
spaces.Box = Box; gym.Env = Env; gym.spaces = spaces
sys.modules["gym"] = gym; sys.modules["gym.spaces"] = spaces
from pathlib import Path
ADAPTER = Path(sys.argv[1]) if len(sys.argv) > 1 else \
    Path(__file__).resolve().parents[1] / "envs" / "wrappers" / "safety_multigoal_joint.py"
spec = importlib.util.spec_from_file_location("adapter", str(ADAPTER))
mod = importlib.util.module_from_spec(spec); spec.loader.exec_module(mod)
Adapter = mod.SafetyMultiGoalJoint

# ---- mock of the Safety-Gymnasium multi-agent API (dict-valued, spaces are methods) ----
class FakeSpace:
    def __init__(self, n, lo=-1.0, hi=1.0):
        self.low = np.full(n, lo, np.float32); self.high = np.full(n, hi, np.float32); self.shape = (n,)
class FakeSafety:
    def __init__(self, T=5, differing_obs=False):
        self.T, self.t, self.differing = T, 0, differing_obs
        self.seen_actions, self.reset_seeds, self.costs_next = [], [], {"agent_0": 0.0, "agent_1": 0.0}
        self.closed = False
    def observation_space(self, a): return FakeSpace(208, -np.inf, np.inf)
    def action_space(self, a): return FakeSpace(8)
    def _obs(self):
        o = np.arange(208, dtype=np.float64) + self.t
        d = {"agent_0": o.copy(), "agent_1": o.copy()}
        if self.differing: d["agent_1"][0] += 1
        return d
    def reset(self, seed=None, options=None):
        self.t = 0; self.reset_seeds.append(seed); return self._obs(), {"agent_0": {}, "agent_1": {}}
    def step(self, acts):
        self.seen_actions.append({k: v.copy() for k, v in acts.items()}); self.t += 1
        trunc = self.t >= self.T
        return (self._obs(), {"agent_0": 1.0, "agent_1": 0.5}, dict(self.costs_next),
                {"agent_0": False, "agent_1": False}, {"agent_0": trunc, "agent_1": trunc}, {})
    def render(self): return "rendered"
    def close(self): self.closed = True

ok = 0
def check(cond, msg):
    global ok
    assert cond, "FAILED: " + msg
    ok += 1; print("ok  -", msg)

fake = FakeSafety(T=3); env = Adapter(level=1, inner_env=fake)
check(env.agents == ["agent_0", "agent_1"], "agent ids taken from the reset observation")
check(env.observation_space.shape == (208,) and env.observation_space.dtype == np.float32, "observation space is 208-d float32")
check(env.action_space.shape == (16,), "joint action space is 16-d (2 x 8)")
check(np.all(env.action_space.low == -1) and np.all(env.action_space.high == 1), "joint action bounds carried over")

obs = env.reset()
check(isinstance(obs, np.ndarray) and obs.shape == (208,) and obs.dtype == np.float32, "reset returns ONE float32 observation (old gym API)")

a = np.concatenate([np.full(8, 0.1), np.full(8, -0.2)])
out = env.step(a)
check(len(out) == 4, "step returns the old 4-tuple")
obs, reward, done, info = out
check(np.allclose(fake.seen_actions[-1]["agent_0"], 0.1) and np.allclose(fake.seen_actions[-1]["agent_1"], -0.2), "joint action is split correctly per agent")
check(abs(reward - 1.5) < 1e-9, "reward is the sum over agents")
check(info["cost"] == 0.0 and info["lag_cost"] == 0, "no cost -> cost 0, lag_cost 0")
check(done is False, "not done before the horizon")

fake.costs_next = {"agent_0": 0.0, "agent_1": 1.0}
_, _, _, info = env.step(a)
check(info["cost"] == 1.0 and info["lag_cost"] == 1, "one agent's cost -> summed cost 1, lag_cost 1")
check(info["cost_agent_0"] == 0.0 and info["cost_agent_1"] == 1.0, "per-agent costs are kept in info")
fake.costs_next = {"agent_0": 1.0, "agent_1": 1.0}
_, _, done, info = env.step(a)
check(info["cost"] == 2.0 and info["lag_cost"] == 1, "both agents violate -> summed cost 2, lag_cost still 1")
check(done is True and info["TimeLimit.truncated"] is True, "truncation -> done, flagged as a time-limit truncation")

env.seed(123); env.reset(); env.reset()
check(fake.reset_seeds[-2:] == [123, None], "seed is passed to the next reset only")

try: env.step(np.zeros(8)); check(False, "wrong action length must raise")
except ValueError: check(True, "wrong action length raises ValueError")

try: Adapter(level=1, inner_env=FakeSafety(differing_obs=True)); check(False, "different observations must raise")
except RuntimeError: check(True, "different per-agent observations raise instead of silently dropping one")

check(env.render() == "rendered", "render is delegated")
env.close(); check(fake.closed, "close is delegated")
print("\nALL", ok, "CHECKS PASSED")
