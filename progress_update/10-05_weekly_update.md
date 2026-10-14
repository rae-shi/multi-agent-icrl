# Env test and ICRL port

## Safe Multi-Agent environments

- The [docs](https://safety-gymnasium.readthedocs.io/en/stable/environments/safe_multi_agent.html) list 8 environments: MultiGoal, Multi-Agent Velocity, 2 FreightFranka, 4 ShadowHand.
- The 6 FreightFranka and ShadowHand tasks are built on Isaac Gym (they import `isaacgym`). MultiGoal and Velocity are not.
- Isaac Gym needs Python 3.8, Ubuntu 18.04/20.04 and a manual NVIDIA download. It would need an Apptainer container, and Preview 4 (2022) is untested on our GPUs (H100 on Fir, A100 on Narval). Not pursued for now.
- Following the documentation, we only use the documented costs and ignore the agent-to-agent contact cost (`cost_contact_other`), which never fires as installed. So the cost is hazards at level 1, plus vases at level 2.
- Multi-Agent Velocity: <mark>TODO</mark>.

## ICRL code port

- Source: official ICRL-benchmarks-public (BSD 3-Clause), upstream commit `0d79d2c`.
- `scripts/port_icrl_code.sh` copies the code into `algos/icrl/`(254 files), keeps the license and records the provenance. 

## Paper

- LaTeX draft started in Overleaf; the Environments section is written.

## Next

- Test on MultiGoal level 1: train the PPO-Lag expert, collect expert data, run ICRL, make plots.