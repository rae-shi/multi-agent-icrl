"""Joint-action adapter: Safety-Gymnasium MultiGoal as an old-gym (0.21) single-agent env.

The ICRL benchmark code expects ``gym`` 0.21 semantics: ``reset()`` returns an observation,
``step()`` returns ``(obs, reward, done, info)`` and the cost is read from ``info``.
Safety-Gymnasium's MultiGoal is a gymnasium multi-agent env: dict-valued ``reset``/``step``,
and ``observation_space`` / ``action_space`` are methods taking an agent id.

This adapter is the "joint-action stub":
  * action      = the agents' actions concatenated (2 x 8 = 16 dims)
  * observation = the shared observation (both agents receive the same one; checked)
  * reward      = sum over agents
  * info['cost']     = sum of the agents' costs (each agent's ``cost_sum``)
  * info['lag_cost'] = 1 if any agent has a cost > 0 (the ground-truth violation flag the
                       ICRL monitor reads)
  * done        = terminated or truncated for any agent

It is a pipeline test, not the independent two-agent setup. Register ids in
``mujuco_environment/custom_envs/__init__.py`` (see algos/patches).
"""
import os

import gym
import numpy as np
from gym import spaces


def _box(low, high):
    return spaces.Box(low=np.asarray(low, dtype=np.float32),
                      high=np.asarray(high, dtype=np.float32),
                      dtype=np.float32)


class SafetyMultiGoalJoint(gym.Env):
    metadata = {"render.modes": []}

    def __init__(self, level=1, inner_env=None):
        if inner_env is None:
            # headless MuJoCo rendering (needed on the cluster; SubprocVecEnv children inherit this)
            os.environ.setdefault("MUJOCO_GL", "egl")
            os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
            import safety_gymnasium  # lazy: only needed for the real environment
            inner_env = safety_gymnasium.make("SafetyAntMultiGoal{0}-v0".format(level))
        self._env = inner_env
        self._env.unwrapped.task.contact_other_cost = 0.0
        self.level = level
        self._seed_for_next_reset = None

        obs, _ = self._env.reset()
        self.agents = list(obs.keys())
        self._check_shared(obs)

        o_space = self._env.observation_space(self.agents[0])
        self.observation_space = _box(o_space.low, o_space.high)

        lows, highs, self._slices, start = [], [], [], 0
        for a in self.agents:
            a_space = self._env.action_space(a)
            low = np.asarray(a_space.low, dtype=np.float32).reshape(-1)
            high = np.asarray(a_space.high, dtype=np.float32).reshape(-1)
            lows.append(low)
            highs.append(high)
            self._slices.append(slice(start, start + low.shape[0]))
            start += low.shape[0]
        self._action_total = start
        self.action_space = _box(np.concatenate(lows), np.concatenate(highs))

    # -- helpers ---------------------------------------------------------------
    def _check_shared(self, obs):
        first = np.asarray(obs[self.agents[0]], dtype=np.float32)
        for a in self.agents[1:]:
            if not np.array_equal(first, np.asarray(obs[a], dtype=np.float32)):
                raise RuntimeError(
                    "agents received different observations; this adapter assumes one shared "
                    "observation and would silently drop information")
        return first

    # -- old gym API -----------------------------------------------------------
    def seed(self, seed=None):
        self._seed_for_next_reset = seed
        return [seed]

    def reset(self):
        if self._seed_for_next_reset is not None:
            out = self._env.reset(seed=self._seed_for_next_reset)
            self._seed_for_next_reset = None  # gymnasium semantics: seed once
        else:
            out = self._env.reset()
        obs = out[0] if isinstance(out, tuple) else out
        return self._check_shared(obs)

    def step(self, action):
        action = np.asarray(action, dtype=np.float32).reshape(-1)
        if action.shape[0] != self._action_total:
            raise ValueError("expected a joint action of length {0}, got {1}".format(
                self._action_total, action.shape[0]))
        joint = {a: action[s] for a, s in zip(self.agents, self._slices)}
        obs, rewards, costs, terminated, truncated, _ = self._env.step(joint)

        obs = self._check_shared(obs)
        reward = float(sum(rewards[a] for a in self.agents))
        agent_costs = [float(costs[a]) for a in self.agents]
        cost = float(sum(agent_costs))
        term = any(bool(terminated[a]) for a in self.agents)
        trunc = any(bool(truncated[a]) for a in self.agents)

        info = {"cost": cost,
                "lag_cost": int(cost > 0),
                "TimeLimit.truncated": bool(trunc and not term)}
        for a, c in zip(self.agents, agent_costs):
            info["cost_" + a] = c
        return obs, reward, bool(term or trunc), info

    def render(self, mode="human"):
        return self._env.render()

    def close(self):
        self._env.close()
