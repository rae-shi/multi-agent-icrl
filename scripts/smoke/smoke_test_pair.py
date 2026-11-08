"""Smoke test of the TWO-ROW adapter on the REAL Safety-Gymnasium MultiGoal (run it as a SLURM job).

From the repo root, with patches 0001-0006 applied:
    export PYTHONPATH="$PWD:$PWD/algos/icrl:$PYTHONPATH"
    python scripts/smoke/smoke_test_pair.py
Part 1: one raw simulation (spaces, one-hot, per-agent violations, contact cost off, speed).
Part 2: the benchmark's own make_train_env (2 simulations -> 4 rows, SubprocVecEnv + cost wrappers +
        VecNormalize): per-row monitor files must match the violations seen step by step.
Part 3: make_eval_env + the patched evaluate_icrl_policy and sampler on 2 rows.
Exits 1 on any failure.
"""
import csv, os, sys, tempfile, time
import numpy as np

ENV_ID = "SafetyAntMultiGoal1Pair-v0"
failures = []
def check(cond, msg):
    print(("ok   - " if cond else "FAIL - ") + msg, flush=True)
    if not cond: failures.append(msg)

rng = np.random.default_rng(0)
ONEHOT = [[1, 0], [0, 1]]

def main():
    # ------------------------------------------------------------------ Part 1: raw simulation ------
    print("== 1. one raw simulation ==", flush=True)
    from envs.wrappers.safety_multigoal_pair import SafetyMultiGoalPair
    env = SafetyMultiGoalPair(level=1)
    check(env.contact_cost_disabled, "agent-to-agent contact cost is switched off on the real task")
    check(env.observation_space.shape == (420,) and env.action_space.shape == (16,), "flat spaces: 420-d observation (2 x 210), 16-d joint action")
    obs = env.reset()
    check([list(obs[208:210]), list(obs[418:420])] == ONEHOT, "one-hot ids at columns 208:210 (row 0) and 418:420 (row 1)")
    check(np.array_equal(obs[:208], obs[210:418]), "both rows carry the same 208-d shared observation")
    N1 = 3000
    viol, both, onehot_ok, t0 = [0, 0], 0, True, time.time()
    episodes, ep_len, ep_v = [], 0, [0, 0]          # per episode: (length, ended by time limit?, violating steps per agent)
    for t in range(N1):
        obs, r, done, info = env.step(rng.uniform(-1, 1, 16).astype(np.float32))
        ep_len += 1
        onehot_ok &= [list(obs[208:210]), list(obs[418:420])] == ONEHOT
        flags = [info["rows"][a]["lag_cost"] for a in range(2)]
        for a in range(2):
            viol[a] += flags[a]; ep_v[a] += flags[a]
        both += int(flags[0] and flags[1])
        if done:
            episodes.append((ep_len, bool(info["TimeLimit.truncated"]), tuple(ep_v)))
            ep_len, ep_v = 0, [0, 0]
            env.reset()
    dt = time.time() - t0
    print(f"   {N1} sim steps in {dt:.1f}s = {N1/dt:.0f} sim steps/s", flush=True)
    if episodes:
        L = sorted(e[0] for e in episodes)
        print(f"   {len(episodes)} episodes finished; length min/median/max = {L[0]}/{L[len(L)//2]}/{L[-1]}; "
              f"ended by the time limit: {sum(e[1] for e in episodes)}, ended early (terminated): {sum(not e[1] for e in episodes)}", flush=True)
        print(f"   episodes with a violation: agent_0={sum(e[2][0] > 0 for e in episodes)}, agent_1={sum(e[2][1] > 0 for e in episodes)} of {len(episodes)}", flush=True)
    print(f"   violating steps: agent_0={viol[0]}, agent_1={viol[1]}, both at once={both}  (random policy; exposure per agent)", flush=True)
    check(onehot_ok, "one-hot ids stay correct on every step")
    if max(viol) > 0:
        check(not (viol[0] == viol[1] == both), "the two agents do not always violate together (costs are per agent)")
    env.close()

    # --------------------------------------------- Part 2: the benchmark's own training factory ------
    print("\n== 2. make_train_env (2 simulations -> 4 rows) ==", flush=True)
    from common.cns_env import make_train_env, make_eval_env
    tmp = tempfile.mkdtemp() + os.sep
    train_env, _ = make_train_env(env_id=ENV_ID, config_path=None, save_dir=tmp, group="PPO-Lag", base_seed=0,
                                  num_threads=2, use_cost=True, multi_env=True, cost_info_str="cost",
                                  reward_gamma=0.99, cost_gamma=0.99)
    check(train_env.num_envs == 4, "2 simulations -> 4 rows")
    obs = train_env.reset()
    check(obs.shape == (4, 210), "per-row observations are 210-d")
    if hasattr(train_env, "get_original_obs"):
        check([list(o[208:210]) for o in train_env.get_original_obs()] == ONEHOT + ONEHOT, "unnormalized one-hots: rows 2s / 2s+1 = agent 0 / agent 1")
    seen, cur, cost_eq, t0 = [[] for _ in range(4)], [0] * 4, True, time.time()
    for t in range(6000):
        obs, rew, dones, infos = train_env.step(rng.uniform(-1, 1, (4, 8)).astype(np.float32))
        for i in range(4):
            if infos[i]["lag_cost"]: cur[i] = 1
            cost_eq &= ((infos[i]["cost"] > 0) == bool(infos[i]["lag_cost"]))
            if dones[i]: seen[i].append(cur[i]); cur[i] = 0
        if min(len(x) for x in seen) >= 2: break
    dt = time.time() - t0
    print(f"   {t+1} vector steps in {dt:.1f}s = {4*(t+1)/dt:.0f} rows/s ({2*(t+1)/dt:.0f} sim steps/s over 2 workers)", flush=True)
    print(f"   normalized info['cost'] is positive exactly when lag_cost = 1: {cost_eq} (informational: VecNormalizeWithCost rescales info['cost'], the raw flag stays in lag_cost)", flush=True)
    check(min(len(x) for x in seen) >= 2, "every row finished at least 2 episodes")
    train_env.close()
    mon = {}
    for r in range(4):
        path = os.path.join(tmp, f"r{r}_monitor.csv")
        check(os.path.exists(path), f"r{r}_monitor.csv written")
        with open(path) as fh:
            mon[r] = list(csv.DictReader(fh.read().splitlines()[1:]))
    for r in range(4):
        n = min(len(mon[r]), len(seen[r]))
        check(n > 0 and [int(x["constraint"]) for x in mon[r][:n]] == seen[r][:n], f"row {r}: monitor 'constraint' column matches the violations seen step by step {seen[r][:n]}")
    for s in range(2):
        n = min(len(mon[2 * s]), len(mon[2 * s + 1]))
        check([x["len"] for x in mon[2 * s][:n]] == [x["len"] for x in mon[2 * s + 1][:n]], f"simulation {s}: its two rows end their episodes together (same lengths)")
    print("   first monitor rows (row 0 / row 1):", mon[0][:2], mon[1][:2], flush=True)

    # ---------------------------- Part 3: evaluation + sampler (patches 0005 / 0006) on 2 rows -------
    print("\n== 3. make_eval_env + patched evaluate_icrl_policy and sampler ==", flush=True)
    eval_env, _ = make_eval_env(env_id=ENV_ID, config_path=None, save_dir=tmp, group="PPO-Lag", num_threads=1,
                                mode="test", use_cost=True, cost_info_str="cost", multi_env=False)
    check(eval_env.num_envs == 2, "evaluation env = 1 simulation = 2 rows")
    class RandPolicy:
        def predict(self, obs, state=None, deterministic=True):
            return rng.uniform(-1, 1, (obs.shape[0], 8)).astype(np.float32), None
    from common.cns_evaluation import evaluate_icrl_policy
    res = evaluate_icrl_policy(RandPolicy(), eval_env, record_info_names=[], n_eval_episodes=2, deterministic=True, cost_info_str="cost")
    print(f"   evaluate_icrl_policy -> mean_reward={res[0]:.2f} mean_nc_reward={res[2]:.2f}", flush=True)
    check(np.isfinite(res[0]) and np.isfinite(res[2]), "evaluate_icrl_policy runs on 2 rows and returns finite numbers")
    from common.cns_sampler import ConstrainedRLSampler
    sampler = ConstrainedRLSampler(rollouts=1, store_by_game=True, cost_info_str="cost", env=eval_env, sample_multi_env=False, env_id=ENV_ID)
    o_orig, o, acs, rs, sums, lens = sampler.sample_from_agent(RandPolicy(), eval_env)
    check(len(o) == 2 and o[0].shape[1] == 210 and acs[0].shape[1] == 8 and o[0].shape[0] == lens[0], "sampler: 1 episode -> 2 trajectories (one per agent), 210-d obs, 8-d actions")
    ids = [[float(v) for v in x[0, 208:210]] for x in o_orig]
    check(ids == [[1.0, 0.0], [0.0, 1.0]] and all((x[:, 208:210] == x[0, 208:210]).all() for x in o_orig),
          f"sampler: trajectory 0 is agent 0, trajectory 1 is agent 1 (original observations; ids seen: {ids})")
    check(all(np.allclose(x[:, 208:210], o_orig[i][:, 208:210], atol=1e-6) for i, x in enumerate(o)),
          "sampler: normalized observations agree with the original ones up to normalization (no row mix-up)")
    eval_env.close()

    print("\nSMOKE TEST " + ("FAILED: " + "; ".join(failures) if failures else "PASSED"), flush=True)
    sys.exit(1 if failures else 0)


if __name__ == "__main__":   # required: SubprocVecEnv workers re-import this file
    main()