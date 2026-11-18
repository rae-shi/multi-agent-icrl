"""Record videos of trained PPO-Lag checkpoints on the two-row MultiGoal env (headless, rgb_array).

Loads each checkpoint exactly like scripts/pipeline/generate_expert_data.py (same eval env, same trained
observation statistics, same model loader), renders the simulation every few steps and writes, per
checkpoint and episode:  <out>/it<N>_ep<k>_<det|sto>.mp4 (or .gif), a strip of the last frames (.png),
and one <out>/summary.csv (length, how the episode ended, return, violating steps per agent).

Run from the repo root on a compute node (module load StdEnv/2023 glfw/3.3.8; venv active):
    export MUJOCO_GL=egl PYOPENGL_PLATFORM=egl
    export PYTHONPATH="$PWD:$PWD/algos/icrl:$PYTHONPATH"
    python scripts/env_checks/record_policy_video.py --run-dir runs/PPO-Lag-MG1-Pair-budget1p0/<run> \
        --iterations 10 20 30 40 --episodes 2 --out videos/budget1p0

Caveat: a VecEnv auto-resets when an episode ends, so the terminal frame is not available; the last
saved frame is the state one step before the end. 'ended' = fell (terminated) or time_limit (truncated).
"""
import argparse, csv, os, sys, tempfile
import numpy as np


def put_label(frame, text):
    from PIL import Image, ImageDraw
    img = Image.fromarray(frame)
    d = ImageDraw.Draw(img)
    d.rectangle([0, 0, img.width, 16], fill=(0, 0, 0))
    d.text((4, 2), text, fill=(255, 255, 255))
    return np.asarray(img)


def find_adapter(env):
    """Walk the wrapper chain down to the object that owns the simulation (has render + _fall_penalty)."""
    seen, stack = set(), [env]
    while stack:
        o = stack.pop()
        if o is None or id(o) in seen:
            continue
        seen.add(id(o))
        if hasattr(o, "_fall_penalty") and hasattr(o, "render"):
            return o
        for name in ("venv", "env", "_env", "unwrapped"):
            c = getattr(o, name, None)
            if c is not None and c is not o:
                stack.append(c)
        for c in (getattr(o, "envs", None) or []):
            stack.append(c)
    raise RuntimeError("could not find the SafetyMultiGoalPair adapter inside the wrapper chain")


def record_episode(env, model, render_fn, obs, deterministic, every, label, max_steps=2000):
    """Run until the simulation ends. Returns (frames, info_dict, obs_for_next_episode)."""
    rows = env.num_envs
    frames, ret, viol, t, done, last_infos = [], np.zeros(rows), np.zeros(rows, int), 0, False, None
    while not done and t < max_steps:
        if t % every == 0:
            fr = render_fn()
            if fr is not None:
                frames.append(put_label(np.asarray(fr), "{0}  step {1}".format(label, t)))
        action, _ = model.predict(obs, state=None, deterministic=deterministic)
        obs, reward, dones, infos = env.step(action)
        for r in range(rows):
            ret[r] += float(reward[r])
            viol[r] += int(infos[r].get("lag_cost", 0))
        t += 1
        done = bool(np.all(dones))
        last_infos = infos
    truncated = bool(last_infos[0].get("TimeLimit.truncated", False)) if last_infos else False
    ended = "time_limit" if (t >= 1000 or truncated or not done) else "fell"
    return frames, {"length": t, "ended": ended, "return_a0": ret[0], "return_a1": ret[1],
                    "viol_a0": int(viol[0]), "viol_a1": int(viol[1])}, obs


def save_clip(frames, path_base, fps):
    import imageio
    try:
        imageio.mimsave(path_base + ".mp4", frames, fps=fps)
        return path_base + ".mp4"
    except Exception as e:  # no ffmpeg plugin -> GIF
        print("mp4 failed ({0}); writing GIF".format(e))
        imageio.mimsave(path_base + ".gif", frames, duration=1.0 / fps)
        return path_base + ".gif"


def save_strip(frames, path, n=8, tail=40):
    from PIL import Image
    tl = frames[-tail:]
    idx = np.linspace(0, len(tl) - 1, min(n, len(tl))).astype(int)
    imgs = [Image.fromarray(tl[i]) for i in idx]
    w, h = imgs[0].size
    cols = 4
    rowsn = (len(imgs) + cols - 1) // cols
    canvas = Image.new("RGB", (cols * w, rowsn * h))
    for k, im in enumerate(imgs):
        canvas.paste(im, ((k % cols) * w, (k // cols) * h))
    canvas.save(path)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--iterations", nargs="+", required=True, help="checkpoint iterations, e.g. 10 20 30 40")
    ap.add_argument("--episodes", type=int, default=2)
    ap.add_argument("--out", required=True)
    ap.add_argument("--env-id", default="SafetyAntMultiGoal1Pair-v0")
    ap.add_argument("--seed", type=int, default=1234)
    ap.add_argument("--stochastic", action="store_true", help="sample actions (default: deterministic mean)")
    ap.add_argument("--every", type=int, default=2, help="keep every k-th simulation step")
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--width", type=int, default=480)
    ap.add_argument("--height", type=int, default=360)
    args = ap.parse_args()

    os.environ.setdefault("MUJOCO_GL", "egl")
    os.environ.setdefault("PYOPENGL_PLATFORM", "egl")
    import safety_gymnasium
    _orig_make = safety_gymnasium.make

    def make_rgb(*a, **kw):  # the adapter calls safety_gymnasium.make without a render mode
        kw.setdefault("render_mode", "rgb_array")
        try:
            return _orig_make(*a, width=args.width, height=args.height, **kw)
        except TypeError:
            return _orig_make(*a, **kw)
    safety_gymnasium.make = make_rgb

    from common.cns_env import make_eval_env
    from stable_baselines3.common import vec_env
    from utils.data_utils import load_ppo_model

    os.makedirs(args.out, exist_ok=True)
    tag = "sto" if args.stochastic else "det"
    rows_out = []
    for it in args.iterations:
        stats = os.path.join(args.run_dir, "model_{0}_itrs".format(it), "train_env_stats.pkl")
        if not os.path.exists(stats):
            print("skip iteration {0}: {1} missing".format(it, stats))
            continue
        eval_env, _ = make_eval_env(env_id=args.env_id, config_path=None, save_dir=tempfile.mkdtemp(),
                                    group="PPO-Lag", num_threads=1, mode="expert", use_cost=True,
                                    cost_info_str="cost", multi_env=False)
        env = vec_env.VecNormalizeWithCost.load(stats, eval_env.venv)
        env.training, env.norm_reward = False, False
        model = load_ppo_model(args.run_dir, it, log_file=None)
        adapter = find_adapter(env)
        env.seed(args.seed)
        obs = env.reset()
        for ep in range(args.episodes):
            label = "it{0} ep{1} {2}".format(it, ep, tag)
            frames, info, obs = record_episode(env, model, adapter.render, obs, not args.stochastic,
                                               args.every, label)
            base = os.path.join(args.out, "it{0}_ep{1}_{2}".format(it, ep, tag))
            if frames:
                path = save_clip(frames, base, args.fps)
                save_strip(frames, base + "_last_frames.png")
            else:
                path = "NO FRAMES (render returned None)"
            info.update({"iteration": it, "episode": ep, "mode": tag, "file": path})
            rows_out.append(info)
            print("it {iteration} ep {episode}: length {length}, ended {ended}, return a0/a1 "
                  "{return_a0:.1f}/{return_a1:.1f}, violations {viol_a0}/{viol_a1} -> {file}".format(**info))
        env.close()

    if rows_out:
        with open(os.path.join(args.out, "summary_{0}.csv".format(tag)), "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows_out[0].keys()))
            w.writeheader()
            w.writerows(rows_out)


if __name__ == "__main__":
    main()