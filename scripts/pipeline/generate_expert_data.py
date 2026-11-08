"""Collect expert demonstrations from a trained PPO-Lag policy on the two-row MultiGoal env.

Writes ONE pickle per trajectory (one trajectory = one agent's row of one simulation episode) in the
format the ICRL benchmark's ``load_expert_data`` reads (keys: original_observations, actions, reward,
reward_sum). Files are named ep0000_a0.pkl, ep0000_a1.pkl, ep0001_a0.pkl ... so that the loader's
sorted order alternates agent 0 / agent 1 and ``expert_rollouts: 2k`` in the ICRL config = k episodes.

Run from the repo root (patches 0001-0006 applied):
    export PYTHONPATH="$PWD:$PWD/algos/icrl:$PYTHONPATH"
    python scripts/pipeline/generate_expert_data.py --run-dir runs/PPO-Lag-MG1-Pair/<run> --iteration best \
        --episodes 20 --out runs/expert_data/MultiGoal1Pair
"""
import argparse, os, pickle, sys, tempfile
import numpy as np

ENV_ID = "SafetyAntMultiGoal1Pair-v0"


def collect_expert_episodes(env, model, n_episodes, deterministic=True, seed=None):
    """Run ``n_episodes`` simulation episodes on ONE simulation (a two-row env, ``env.num_envs`` rows).

    ``env`` must provide get_original_obs() (a VecNormalize wrapper does) and report the 0/1
    ``lag_cost`` flag in info. Returns a list of per-row trajectory dicts, in the order
    episode 0 row 0, episode 0 row 1, episode 1 row 0, ...
    """
    rows = env.num_envs
    if seed is not None:
        env.seed(seed)
    trajectories = []
    obs = env.reset()   # a VecEnv auto-resets: after `done`, the returned obs already starts the next episode
    for ep in range(n_episodes):
        orig, acts, rews, viol = ([[] for _ in range(rows)] for _ in range(4))
        done = False
        while not done:
            orig_obs = env.get_original_obs()
            action, _ = model.predict(obs, state=None, deterministic=deterministic)
            for r in range(rows):
                orig[r].append(np.array(orig_obs[r], dtype=np.float32))
                acts[r].append(np.array(action[r], dtype=np.float32))
            obs, reward, dones, infos = env.step(action)
            for r in range(rows):
                rews[r].append(float(reward[r]))
                viol[r].append(int(infos[r]["lag_cost"]))
            done = bool(np.all(dones))
        for r in range(rows):
            trajectories.append({"original_observations": np.stack(orig[r]),
                                 "actions": np.stack(acts[r]),
                                 "reward": np.asarray(rews[r], dtype=np.float32),
                                 "reward_sum": float(np.sum(rews[r])),
                                 "lag_cost": np.asarray(viol[r], dtype=np.int8),
                                 "episode": ep, "agent": r})
    return trajectories


def save_trajectories(trajectories, out_dir):
    os.makedirs(out_dir, exist_ok=True)
    for t in trajectories:
        with open(os.path.join(out_dir, "ep{0:04d}_a{1}.pkl".format(t["episode"], t["agent"])), "wb") as fh:
            pickle.dump(t, fh)


def summarize(trajectories):
    lines = []
    for agent in sorted({t["agent"] for t in trajectories}):
        ts = [t for t in trajectories if t["agent"] == agent]
        lines.append("agent_{0}: {1} episodes | mean return {2:.2f} | mean length {3:.0f} | violating steps/episode "
                     "{4:.1f} | episodes with a violation {5}/{1}".format(
                         agent, len(ts), np.mean([t["reward_sum"] for t in ts]), np.mean([len(t["actions"]) for t in ts]),
                         np.mean([t["lag_cost"].sum() for t in ts]), sum(int(t["lag_cost"].any()) for t in ts)))
    return lines


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True, help="training run folder (contains best_nominal_model.zip / model_<i>_itrs)")
    ap.add_argument("--iteration", default="best", help="'best' or an iteration number (model_<i>_itrs)")
    ap.add_argument("--episodes", type=int, default=20, help="simulation episodes (each gives one trajectory per agent)")
    ap.add_argument("--out", required=True, help="output folder = expert_path of the ICRL config")
    ap.add_argument("--env-id", default=ENV_ID)
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--stochastic", action="store_true", help="sample actions instead of the deterministic mean")
    ap.add_argument("--force", action="store_true", help="write into a non-empty output folder")
    args = ap.parse_args()

    if os.path.isdir(args.out) and os.listdir(args.out) and not args.force:
        sys.exit("refusing to write into non-empty {0} (use --force)".format(args.out))

    from common.cns_env import make_eval_env
    from stable_baselines3.common import vec_env
    from utils.data_utils import load_ppo_model

    stats = os.path.join(args.run_dir, "train_env_stats.pkl") if args.iteration == "best" else \
        os.path.join(args.run_dir, "model_{0}_itrs".format(args.iteration), "train_env_stats.pkl")
    if not os.path.exists(stats):
        sys.exit("normalization statistics not found: {0}".format(stats))
    tmp = tempfile.mkdtemp()
    eval_env, _ = make_eval_env(env_id=args.env_id, config_path=None, save_dir=tmp, group="PPO-Lag", num_threads=1,
                                mode="expert", use_cost=True, cost_info_str="cost", multi_env=False)
    # same wrapper chain as the evaluation env, but with the TRAINED observation statistics
    env = vec_env.VecNormalizeWithCost.load(stats, eval_env.venv)
    env.training = False          # do not update the statistics while collecting
    env.norm_reward = False       # store raw rewards
    model = load_ppo_model(args.run_dir, args.iteration, log_file=None)

    trajectories = collect_expert_episodes(env, model, args.episodes, deterministic=not args.stochastic, seed=args.seed)
    save_trajectories(trajectories, args.out)
    print("wrote {0} trajectories ({1} episodes x {2} agents) to {3}".format(
        len(trajectories), args.episodes, env.num_envs, args.out))
    for line in summarize(trajectories):
        print(line)
    env.close()


if __name__ == "__main__":
    main()
