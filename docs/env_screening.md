Env Screening Results

Screened four third-party PettingZoo candidates against PettingZoo's own
conformance tests (`api_test` for AEC, `parallel_api_test` for Parallel).
All results below are in Python 3.12 venv with `pettingzoo==1.27.0`.

## Summary table

| Candidate | Installs cleanly? | Conformance |
|---|---|---|
| **DSSE** (Drone Swarm Search) | Yes, clean pip install | ✅ PASS (Parallel API) |
| **POGEMA** | Yes (`pip install pogema`) | ✅ PASS (Parallel API), needs its own isolated venv | 
| **Box Jump** | Needs system `swig` binary + Box2D (not pip-only) — confirmed working once `swig` installed | ✅ PASS (Parallel API) | 
| **PettingZoo Dilemmas** (mrochk) | Installs via pip, but wheel is **empty** (packaging bug); works via `PYTHONPATH` from source | ✅ PASS (Parallel API)|
| **PettingZoo Dilemma Envs** (tianyu-z) | Not pip-installable (no `setup.py`/`pyproject.toml`); works via `PYTHONPATH` | ❌ FAIL — after the known `agent_selector` fix, hits a separate, deeper bug: `AssertionError('Out of bounds observation: [2 2]')`, in the repo's own game logic, not yet root-caused |

## Not yet screened in detail
Safety-Gymnasium's native multi-agent envs and Isaac Gym
