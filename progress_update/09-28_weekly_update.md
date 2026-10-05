# Env test 
## Safety-Gymnasium (Safe Multi-Agent)

Install, from source on Python 3.10 (`scripts/setup_safety_gym_env.sh`), needed three workarounds on Alliance:

1. `pygame==2.1.0` cannot build (missing system headers); pin relaxed, pip took 2.5.2.
2. Alliance's pip rejects `manylinux` wheels, so mujoco 2.3.3 tried to compile; installed the manylinux wheel directly after renaming its platform tag.
3. `MUJOCO_GL=egl` is needed for headless import.

What the environment looks like (from running it):

- keyed `agent_0` / `agent_1`; 208-d observation and an 8-d action per agent.
- `step` returns `(obs, rewards, costs, terminated, truncated, infos)`; `costs[agent]` is a per-agent float.
- Both agents receive the same observation, so each can see the other.
- Level 1 has `cost_hazards`; level 2 adds `cost_vases_contact` and `cost_vases_velocity`. These are computed per agent from its own contacts, so they are independent constraints. Under random actions, cost fires on roughly 10-12% of steps at level 2.

Joint constraint: both levels set `contact_other_cost = 1.0` (an agent-to-agent contact cost), but it never fires.

- The cost code matches contacts using `body_info[0].geom_names` and `body_info[1].geom_names`. Both lists contain only geoms of ant 0 (body `agent`); ant 1 is body `agent1` with `1`-suffixed geoms. So the condition cannot match an ant 0 to ant 1 contact.
- The ants can physically collide: all 169 geom pairs are allowed, and teleporting ant 1 onto ant 0 produces real contacts. The geoms are small, so contact needs near-total overlap (none at 0.3 offset).
- Random policies never touch (closest approach 1.0 to 2.3 between ants over a few thousand steps).
