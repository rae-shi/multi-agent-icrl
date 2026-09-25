# Old-stack baseline
Successfully runned [ICRL-benchmarks-public](https://github.com/Guiliang/ICRL-benchmarks-public) GACL/PPO/PPO-Lag algos on Blocked HalfCheetah as the reference. 

# Stable baseline difference
Diffed the `stable_baseline3` bundled inside
[ICRL-benchmarks-public](https://github.com/Guiliang/ICRL-benchmarks-public) against vanilla upstream `v0.9.0`. See results in [sb3_diff](../docs/sb3_diff) folder.

# Env test
Tested candidate [pettingzoo's third-party multi-agent environments](https://pettingzoo.farama.org/environments/third_party_envs/) including DSSE, Box Jump, PettingZoo Dilemmas, POGEMA. See results in [env_screening](../docs/env_screening.md).

