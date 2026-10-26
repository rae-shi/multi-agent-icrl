"""Two-row (per-agent) adapter: Safety-Gymnasium MultiGoal for the ICRL benchmark code.

Design (decided with the professor): independent per-agent constraints, ONE shared policy.
Both agents get the same action space, observation space, reward function and cost function
(only the goal colour differs), so one network can act for both; a 2-d one-hot agent id is
appended to the observation so it knows which agent it is acting for.

Two layers, so that the old benchmark code (SubprocVecEnv / DummyVecEnv / cost wrappers /
PPO-Lag) is reused unchanged:

  1. ``SafetyMultiGoalPair`` (gym.Env, old gym 0.21 API) = ONE simulation.
       action      16-d  = agent_0's 8-d action, then agent_1's 8-d action
       observation 420-d = row 0 (208-d shared obs + [1,0]) then row 1 (208-d shared obs + [0,1])
       reward      scalar (sum; unused) -- the per-row values are in ``info['rows']``
       info['rows'][a] = {'reward', 'cost', 'lag_cost'} of agent a, ``cost`` = that agent's
                         own ``cost_sum`` (hazards only; agent-to-agent contact cost is switched off)
       done        True when the simulation ends (both agents end together)

  2. ``TwoRowVecEnv`` (VecEnvWrapper) turns n simulations into 2n rows (row 2s = agent 0 of
     simulation s, row 2s+1 = agent 1). It reports per-row obs/reward/done/info, and writes one
     monitor file per row (``r<row>_monitor.csv``, same columns as CNSMonitor).

Note: PPO counts a "timestep" per ROW, so one simulation step = 2 timesteps.
"""
import csv
import json
import os
import time

import gym
import numpy as np
from gym import spaces

N_AGENTS = 2
ID_DIM = N_AGENTS                      # one-hot agent id appended to every row's observation
PAIR_SUFFIX = "Pair-v0"
MONITOR_FIELDS = ("reward", "reward_nc", "len", "time", "constraint")


def is_pair_env_id(env_id):
    return isinstance(env_id, str) and "MultiGoal" in env_id and env_id.endswith(PAIR_SUFFIX)


def level_from_env_id(env_id):
    """'SafetyAntMultiGoal1Pair-v0' -> 1"""
    return int(env_id.split("MultiGoal")[1].replace(PAIR_SUFFIX, ""))


def _box(low, high):
    return spaces.Box(low=np.asarray(low, dtype=np.float32),
                      high=np.asarray(high, dtype=np.float32),
                      dtype=np.float32)


class SafetyMultiGoalPair(gym.Env):
    """One Safety-Gymnasium MultiGoal simulation, flattened for the old-gym vector envs."""
    metadata = {"render.modes": []}

    def __init__(self, level=1, inner_env=None):
        created_here = inner_env is None
        if created_here:
            # headless MuJoCo rendering (needed on the cluster; subprocess workers inherit this)
            os.environ.setdefault("MUJOCO_GL", "egl")
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            import safety_gymnasium  # lazy: only needed for the real environment
            inner_env = safety_gymnasium.make("SafetyAntMultiGoal{0}-v0".format(level))
        self._env = inner_env
        self.level = level
        self._seed_for_next_reset = None
        self._require_contact_switch = created_here

        obs, _ = self._env.reset()
        self._disable_agent_contact_cost()
        self.agents = list(obs.keys())
        if len(self.agents) != N_AGENTS:
            raise ValueError("expected {0} agents, got {1}".format(N_AGENTS, self.agents))
        base = self._check_shared(obs)
        self.obs_dim = int(base.shape[0])
        self.row_dim = self.obs_dim + ID_DIM

        # --- spaces; the shared-policy assumption requires both agents' spaces to be identical ---
        o_spaces = [self._env.observation_space(a) for a in self.agents]
        o_low = np.asarray(o_spaces[0].low, dtype=np.float32).reshape(-1)
        o_high = np.asarray(o_spaces[0].high, dtype=np.float32).reshape(-1)
        for sp in o_spaces[1:]:
            if not (np.array_equal(o_low, np.asarray(sp.low, dtype=np.float32).reshape(-1))
                    and np.array_equal(o_high, np.asarray(sp.high, dtype=np.float32).reshape(-1))):
                raise ValueError("agents have different observation spaces: a shared policy is not valid")
        row_low = np.concatenate([o_low, np.zeros(ID_DIM, np.float32)])
        row_high = np.concatenate([o_high, np.ones(ID_DIM, np.float32)])
        self.observation_space = _box(np.tile(row_low, N_AGENTS), np.tile(row_high, N_AGENTS))

        a_lows, a_highs, self._slices, start = [], [], [], 0
        for a in self.agents:
            sp = self._env.action_space(a)
            lo = np.asarray(sp.low, dtype=np.float32).reshape(-1)
            hi = np.asarray(sp.high, dtype=np.float32).reshape(-1)
            a_lows.append(lo)
            a_highs.append(hi)
            self._slices.append(slice(start, start + lo.shape[0]))
            start += lo.shape[0]
        for lo, hi in zip(a_lows[1:], a_highs[1:]):
            if not (np.array_equal(a_lows[0], lo) and np.array_equal(a_highs[0], hi)):
                raise ValueError("agents have different action spaces: a shared policy is not valid")
        self._action_total = start
        self.action_space = _box(np.concatenate(a_lows), np.concatenate(a_highs))

    # -- helpers ------------------------------------------------------------------------------
    def _disable_agent_contact_cost(self):
        """Professor's decision: only the documented costs; drop agent-to-agent contact cost."""
        task = getattr(getattr(self._env, "unwrapped", self._env), "task", None)
        if task is not None and hasattr(task, "contact_other_cost"):
            task.contact_other_cost = 0.0
            self.contact_cost_disabled = True
        else:
            self.contact_cost_disabled = False
            if self._require_contact_switch:
                raise RuntimeError("could not find task.contact_other_cost on the Safety-Gymnasium "
                                   "environment; the excluded contact cost would stay switched on")

    @staticmethod
    def _check_shared(obs):
        agents = list(obs.keys())
        first = np.asarray(obs[agents[0]], dtype=np.float32)
        for a in agents[1:]:
            if not np.array_equal(first, np.asarray(obs[a], dtype=np.float32)):
                raise RuntimeError("agents received different observations; the one-hot design assumes "
                                   "one shared observation and would silently drop information")
        return first

    def _rows(self, base):
        rows = np.zeros((N_AGENTS, self.row_dim), dtype=np.float32)
        rows[:, :self.obs_dim] = base
        rows[np.arange(N_AGENTS), self.obs_dim + np.arange(N_AGENTS)] = 1.0
        return rows.reshape(-1)

    # -- old gym API ----------------------------------------------------------------------------
    def seed(self, seed=None):
        self._seed_for_next_reset = seed
        return [seed]

    def reset(self):
        if self._seed_for_next_reset is not None:
            out = self._env.reset(seed=self._seed_for_next_reset)
            self._seed_for_next_reset = None  # gymnasium semantics: seed once
        else:
            out = self._env.reset()
        self._disable_agent_contact_cost()   # after the reset, so it cannot silently revert
        obs = out[0] if isinstance(out, tuple) else out
        return self._rows(self._check_shared(obs))

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self._action_total:
            raise ValueError("expected a joint action of length {0}, got {1}".format(
                self._action_total, action.shape[0]))
        joint = {a: action[s] for a, s in zip(self.agents, self._slices)}
        obs, rewards, costs, terminated, truncated, _ = self._env.step(joint)

        flat = self._rows(self._check_shared(obs))
        rows = []
        for a in self.agents:
            c = float(costs[a])
            rows.append({"reward": float(rewards[a]), "cost": c, "lag_cost": int(c > 0)})
        term = any(bool(terminated[a]) for a in self.agents)
        trunc = any(bool(truncated[a]) for a in self.agents)
        info = {"rows": rows,
                "cost": float(sum(r["cost"] for r in rows)),
                "lag_cost": int(any(r["lag_cost"] for r in rows)),
                "TimeLimit.truncated": bool(trunc and not term)}
        return flat, float(sum(r["reward"] for r in rows)), bool(term or trunc), info

    def render(self, mode="human"):
        return self._env.render()

    def close(self):
        self._env.close()


# ================================================================================================
from stable_baselines3.common.vec_env.base_vec_env import VecEnvWrapper  # noqa: E402


class TwoRowVecEnv(VecEnvWrapper):
    """n simulations -> 2n rows (row 2s = agent 0 of simulation s, row 2s+1 = agent 1)."""

    def __init__(self, venv, env_id="", log_dir=None):
        inner_obs, inner_act = venv.observation_space, venv.action_space
        self.row_dim = int(inner_obs.shape[0]) // N_AGENTS
        self.act_dim = int(inner_act.shape[0]) // N_AGENTS
        super().__init__(venv,
                         observation_space=_box(inner_obs.low[:self.row_dim], inner_obs.high[:self.row_dim]),
                         action_space=_box(inner_act.low[:self.act_dim], inner_act.high[:self.act_dim]))
        self.n_sims = venv.num_envs
        self.num_envs = N_AGENTS * self.n_sims
        self.env_id = env_id
        self.log_dir = log_dir
        self._t_start = time.time()
        self._files, self._writers = [], []
        self._clear_rows()
        if log_dir is not None:
            self._open_monitors()

    # -- per-row monitor (same semantics and columns as CNSMonitor) ----------------------------
    def _clear_rows(self):
        self._rewards = [[] for _ in range(self.num_envs)]
        self._rewards_nc = [[] for _ in range(self.num_envs)]   # rewards before the first violation
        self._broken = [0] * self.num_envs

    def _monitor_path(self, row):
        if self.log_dir.endswith(os.sep) or os.path.isdir(self.log_dir):
            return os.path.join(self.log_dir, "r{0}_monitor.csv".format(row))
        return self.log_dir + ".r{0}.monitor.csv".format(row)

    def _open_monitors(self):
        for row in range(self.num_envs):
            path = self._monitor_path(row)
            os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
            fh = open(path, "wt")
            fh.write("#%s\n" % json.dumps({"t_start": self._t_start, "env_id": self.env_id}))
            writer = csv.DictWriter(fh, fieldnames=MONITOR_FIELDS, delimiter=",")
            writer.writeheader()
            fh.flush()
            self._files.append(fh)
            self._writers.append(writer)

    def _monitor_step(self, row, reward, lag_cost):
        if lag_cost:                                   # flag first, like CNSMonitor: the violating
            self._broken[row] = 1                      # step's own reward is excluded from reward_nc
        self._rewards[row].append(reward)
        if not self._broken[row]:
            self._rewards_nc[row].append(reward)

    def _monitor_end_episode(self, row):
        ep_info = {"reward": round(float(sum(self._rewards[row])), 2),
                   "reward_nc": round(float(sum(self._rewards_nc[row])), 2),
                   "len": len(self._rewards[row]),
                   "time": round(time.time() - self._t_start, 2),
                   "constraint": self._broken[row]}
        if self._writers:
            self._writers[row].writerow(ep_info)
            self._files[row].flush()
        self._rewards[row], self._rewards_nc[row], self._broken[row] = [], [], 0
        return ep_info

    # -- VecEnv API --------------------------------------------------------------------------------
    def _split_obs(self, flat):
        return np.asarray(flat, dtype=np.float32).reshape(self.num_envs, self.row_dim)

    def reset(self):
        flat = self.venv.reset()
        self._clear_rows()
        return self._split_obs(flat)

    def step_async(self, actions):
        actions = np.asarray(actions, dtype=np.float32)
        if actions.shape != (self.num_envs, self.act_dim):
            raise ValueError("expected actions of shape {0}, got {1}".format(
                (self.num_envs, self.act_dim), actions.shape))
        self.venv.step_async(actions.reshape(self.n_sims, N_AGENTS * self.act_dim))

    def step_wait(self):
        flat, _, sim_dones, sim_infos = self.venv.step_wait()
        obs = self._split_obs(flat)
        rewards = np.zeros(self.num_envs, dtype=np.float32)
        dones = np.repeat(np.asarray(sim_dones, dtype=bool), N_AGENTS)
        infos = []
        for s in range(self.n_sims):
            sim_info = sim_infos[s]
            terminal = None
            if sim_dones[s] and "terminal_observation" in sim_info:
                terminal = np.asarray(sim_info["terminal_observation"], dtype=np.float32).reshape(
                    N_AGENTS, self.row_dim)
            for a in range(N_AGENTS):
                row = N_AGENTS * s + a
                ri = sim_info["rows"][a]
                rewards[row] = ri["reward"]
                info = {"cost": ri["cost"], "lag_cost": ri["lag_cost"],
                        "TimeLimit.truncated": sim_info.get("TimeLimit.truncated", False)}
                self._monitor_step(row, ri["reward"], ri["lag_cost"])
                if sim_dones[s]:
                    if terminal is not None:
                        info["terminal_observation"] = terminal[a]
                    info["episode"] = self._monitor_end_episode(row)
                infos.append(info)
        return obs, rewards, dones, infos

    def close(self):
        for fh in self._files:
            fh.close()
        self._files, self._writers = [], []
        self.venv.close()


def make_pair_vec_env(env_id, num_threads=1, base_seed=0, log_dir=None, parallel=True,
                      env_factory=None, seed_fn=None, start_method=None):
    """Build the two-row vector env. ``parallel`` = SubprocVecEnv (training), else DummyVecEnv."""
    from stable_baselines3.common import vec_env as sb3_vec_env
    level = level_from_env_id(env_id)
    if seed_fn is None:
        from stable_baselines3.common.utils import set_random_seed as seed_fn
    seed_fn(base_seed)

    def make(rank):
        def _init():
            env = env_factory() if env_factory is not None else SafetyMultiGoalPair(level=level)
            env.seed(base_seed + rank)
            return env
        return _init

    fns = [make(i) for i in range(num_threads)]
    if parallel:
        inner = sb3_vec_env.SubprocVecEnv(fns, start_method=start_method)
    else:
        inner = sb3_vec_env.DummyVecEnv(fns)
    return TwoRowVecEnv(inner, env_id=env_id, log_dir=log_dir)
