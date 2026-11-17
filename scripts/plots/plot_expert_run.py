#!/usr/bin/env python
"""Slide-ready plots for one PPO-Lag expert run on the two-row MultiGoal env.

Inputs (all produced by the repo's own scripts, nothing is hand-copied):
  --run-dir   training run folder holding r<i>_monitor.csv   (row i = 2*simulation + agent)
  --log       the job's .out file (per-iteration average_cost, evaluation mean_reward, multiplier nu)
  --eval-glob folders written by scripts/pipeline/generate_expert_data.py, one per checkpoint,
              e.g. "runs/expert_data_check/s123_it*_n30"  (iteration is read from the folder name)
Outputs: three PNGs in --out and, with --csv, the numbers behind them.

Example (from the repo root):
  python scripts/plots/plot_expert_run.py \
      --run-dir runs/PPO-Lag-MG1-Pair/<run> --log expert_ppo_lag_<job>_0.out \
      --eval-glob "runs/expert_data_check/s123_it*_n30" \
      --out progress_update/figures --csv docs/results --tag s123_lr3e-5
"""
import argparse, glob, os, pickle, re, sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

COLORS = {0: "#1f77b4", 1: "#d95f02"}
plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False,
                     "axes.grid": True, "grid.alpha": 0.25, "figure.dpi": 100})


# ----------------------------------------------------------------------------- loading
def load_training_episodes(run_dir):
    parts = []
    for p in glob.glob(os.path.join(run_dir, "r*_monitor.csv")):
        m = re.search(r"r(\d+)_monitor\.csv$", p)
        if not m:
            continue
        idx = int(m.group(1))
        df = pd.read_csv(p, skiprows=1)          # first line is a '#{...}' json header
        parts.append(df.assign(row=idx, agent=idx % 2, sim=idx // 2))
    if not parts:
        sys.exit("no r*_monitor.csv files in " + run_dir)
    return pd.concat(parts, ignore_index=True)


def bin_training(eps, n_bins, max_len):
    """Bins hold equal numbers of simulation steps, so 'progress' means progress through training."""
    out = []
    for agent, g in eps.groupby("agent"):
        g = g.sort_values("time").reset_index(drop=True)
        cum = g["len"].cumsum().to_numpy()
        frac = (cum - g["len"].to_numpy() / 2) / cum[-1]
        g["bin"] = np.minimum((frac * n_bins).astype(int), n_bins - 1)
        for b, h in g.groupby("bin"):
            out.append(dict(agent=agent, bin=int(b), progress=(b + 0.5) / n_bins, episodes=len(h),
                            mean_len=h["len"].mean(), full_len_share=(h["len"] >= max_len).mean(),
                            viol_ep_share=(h["constraint"] > 0).mean()))
    return pd.DataFrame(out)


def parse_log(path):
    """Per-iteration average_cost and evaluation mean_reward from the stats tables, and the multiplier
    nu as last seen in the progress bar before each iteration's table (assumes the bar precedes the table)."""
    text = open(path, errors="ignore").read()
    cost, rew, nu, last_nu = [], [], [], np.nan
    for chunk in re.split(r"[\r\n]", text):
        m = re.search(r"nu=\s*([-\d.]+)", chunk)
        if m:
            last_nu = float(m.group(1))
        m = re.search(r"\|\s+average_cost\s+\|\s+([-\d.eE+]+)\s+\|", chunk)
        if m:
            cost.append(float(m.group(1)))
        m = re.search(r"\|\s+mean_reward\s+\|\s+([-\d.eE+]+)\s+\|", chunk)
        if m:
            rew.append(float(m.group(1)))
        if re.search(r"\|\s+run_iter\s+\|", chunk):
            nu.append(last_nu)
    return cost, rew, nu


def load_checkpoints(pattern, regex, max_len):
    rows = []
    for d in sorted(glob.glob(pattern)):
        m = re.search(regex, os.path.basename(d))
        if not m:
            continue
        for a in (0, 1):
            files = sorted(glob.glob(os.path.join(d, "ep*_a%d.pkl" % a)))
            if not files:
                continue
            ts = [pickle.load(open(f, "rb")) for f in files]   # only load pickles you wrote yourself
            L = np.array([len(t["actions"]) for t in ts])
            V = np.array([int(np.sum(t["lag_cost"])) for t in ts])
            R = np.array([t["reward_sum"] for t in ts])
            rows.append(dict(iteration=int(m.group(1)), agent=a, episodes=len(ts), mean_len=L.mean(),
                             full_len_share=float((L >= max_len).mean()), viol_per_step=V.sum() / L.sum(),
                             eps_with_viol=float((V > 0).mean()), mean_return=R.mean()))
    return pd.DataFrame(rows).sort_values(["iteration", "agent"]).reset_index(drop=True)


# ----------------------------------------------------------------------------- plotting
def smooth(x, k=5):
    x = np.asarray(x, float)
    return np.convolve(x, np.ones(k) / k, mode="valid") if len(x) >= k else x


def fig_training(cost, rew, nu, out, title, budget=0.0):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    it = np.arange(len(cost))
    ax[0].plot(it, np.array(cost) * 100, color="#999", lw=1, marker="o", ms=3, label="per iteration")
    if len(cost) >= 5:
        ax[0].plot(np.arange(len(smooth(cost))) + 2, smooth(cost) * 100, color="#1f77b4", lw=2.2, label="5-iteration mean")
    if budget < 1:
        ax[0].axhline(100 * budget, color="k", lw=1, ls="--"); ax[0].text(0.98, 0.10, f"budget = {100 * budget:g}%", transform=ax[0].transAxes, ha="right", fontsize=9)
    else:
        ax[0].text(0.98, 0.10, "unconstrained (budget = 100%)", transform=ax[0].transAxes, ha="right", fontsize=9)
    ax[0].set_title("Training cost (violating rows)"); ax[0].set_xlabel("iteration"); ax[0].set_ylabel("% of rows with a violation")
    ax[0].set_ylim(bottom=-0.5); ax[0].legend(frameon=False, fontsize=9)
    ax[1].plot(np.arange(len(nu)), nu, color="#d95f02", lw=2)
    ax[1].set_title("Lagrange multiplier"); ax[1].set_xlabel("iteration"); ax[1].set_ylabel("nu")
    ax[2].plot(np.arange(len(rew)), rew, color="#999", lw=1, marker="o", ms=3, label="per iteration")
    if len(rew) >= 5:
        ax[2].plot(np.arange(len(smooth(rew))) + 2, smooth(rew), color="#2ca02c", lw=2.2, label="5-iteration mean")
    ax[2].axhline(0, color="k", lw=0.8)
    ax[2].set_title("Evaluation return (log: mean_reward)"); ax[2].set_xlabel("iteration"); ax[2].set_ylabel("episode return")
    ax[2].legend(frameon=False, fontsize=9, loc="upper left")
    fig.suptitle(title, y=1.02, fontsize=12); fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_lengths(bins, n_bins, max_len, out, title):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
    b0 = bins[bins.agent == 0]
    ax[0].plot(b0.progress * 100, b0.mean_len, color="#1f77b4", lw=2, marker="o", ms=4)
    ax[0].set_title("Mean episode length"); ax[0].set_ylabel("steps"); ax[0].set_ylim(0, max_len * 1.05)
    ax[1].plot(b0.progress * 100, b0.full_len_share * 100, color="#9467bd", lw=2, marker="o", ms=4)
    ax[1].set_title("Episodes reaching the %d-step cap" % max_len); ax[1].set_ylabel("% of episodes")
    for a in (0, 1):
        g = bins[bins.agent == a]
        ax[2].plot(g.progress * 100, g.viol_ep_share * 100, color=COLORS[a], lw=2, marker="o", ms=4, label="agent %d" % a)
    ax[2].set_title("Episodes with a violation"); ax[2].set_ylabel("% of episodes"); ax[2].legend(frameon=False)
    for x in ax:
        x.set_xlabel("training progress (% of simulation steps)")
    fig.suptitle(title + "  [%d equal-step bins]" % n_bins, y=1.02, fontsize=12)
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


def fig_checkpoints(ck, ref, out, title, caption):
    fig, ax = plt.subplots(1, 3, figsize=(13, 3.8))
    its = sorted(ck.iteration.unique())
    for a in (0, 1):
        g = ck[ck.agent == a]
        ax[0].plot(g.iteration, g.mean_len, color=COLORS[a], marker="o", lw=2, label="agent %d" % a) if a == 0 else None
        ax[1].plot(g.iteration, g.viol_per_step * 100, color=COLORS[a], marker="o", lw=2, label="agent %d" % a)
        ax[2].plot(g.iteration, g.mean_return, color=COLORS[a], marker="o", lw=2, label="agent %d" % a)
        if ref and a < len(ref):
            ax[1].axhline(ref[a] * 100, color=COLORS[a], ls="--", lw=1, alpha=0.7, label="random policy, agent %d" % a)
    ax[0].set_title("Mean episode length (both agents)"); ax[0].set_ylabel("steps")
    ax[1].set_title("Violations per step"); ax[1].set_ylabel("% of steps"); ax[1].set_ylim(bottom=0); ax[1].legend(frameon=False, fontsize=9)
    ax[2].set_title("Mean episode return"); ax[2].set_ylabel("return"); ax[2].axhline(0, color="k", lw=0.8)
    for x in ax:
        x.set_xlabel("checkpoint (iteration)"); x.set_xticks(its)
    fig.suptitle(title, y=1.02, fontsize=12)
    if caption:
        fig.text(0.5, -0.04, caption, ha="center", fontsize=9, color="#555")
    fig.tight_layout(); fig.savefig(out, dpi=200, bbox_inches="tight"); plt.close(fig)


# ----------------------------------------------------------------------------- main
def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", required=True)
    ap.add_argument("--log", required=True)
    ap.add_argument("--eval-glob", required=True)
    ap.add_argument("--eval-regex", default=r"_it(\d+)_", help="regex with one group that extracts the iteration from a folder name")
    ap.add_argument("--out", default="progress_update/figures")
    ap.add_argument("--csv", default=None, help="folder for the CSVs behind the figures")
    ap.add_argument("--tag", default="expert", help="file-name prefix and figure title")
    ap.add_argument("--budget", type=float, default=0.0, help="cost budget used in training, as a fraction (0.054 = 5.4%%)")
    ap.add_argument("--bins", type=int, default=20)
    ap.add_argument("--max-len", type=int, default=1000, help="episode cap (num_steps)")
    ap.add_argument("--random-ref", type=float, nargs=2, default=[0.048, 0.167], metavar=("A0", "A1"),
                    help="per-step violation rate of the random policy per agent (set to -1 -1 to hide)")
    ap.add_argument("--caption", default="", help="footnote under the checkpoint figure, e.g. '30 episodes, 1 seed'")
    a = ap.parse_args()
    os.makedirs(a.out, exist_ok=True)
    if a.csv:
        os.makedirs(a.csv, exist_ok=True)
    ref = None if a.random_ref[0] < 0 else a.random_ref

    cost, rew, nu = parse_log(a.log)
    print("log: %d cost values, %d reward values, %d multiplier values" % (len(cost), len(rew), len(nu)))
    eps = load_training_episodes(a.run_dir)
    bins = bin_training(eps, a.bins, a.max_len)
    ck = load_checkpoints(a.eval_glob, a.eval_regex, a.max_len)
    print("training episodes: %d (agent 0: %d) | checkpoints: %s" % (len(eps), (eps.agent == 0).sum(), sorted(ck.iteration.unique())))

    fig_training(cost, rew, nu, os.path.join(a.out, a.tag + "_training_curves.png"), a.tag + f": PPO-Lag training ({'unconstrained' if a.budget >= 1 else 'budget %g%%' % (100 * a.budget)})", a.budget)
    fig_lengths(bins, a.bins, a.max_len, os.path.join(a.out, a.tag + "_episode_lengths.png"), a.tag + ": episode lengths during training")
    fig_checkpoints(ck, ref, os.path.join(a.out, a.tag + "_checkpoints.png"), a.tag + ": checkpoint evaluation", a.caption)
    if a.csv:
        n = max(len(cost), len(rew), len(nu))
        pad = lambda v: list(v) + [np.nan] * (n - len(v))
        pd.DataFrame(dict(iteration=range(n), average_cost=pad(cost), eval_mean_reward=pad(rew), nu=pad(nu))).to_csv(
            os.path.join(a.csv, a.tag + "_iteration_log.csv"), index=False)
        bins.to_csv(os.path.join(a.csv, a.tag + "_training_bins.csv"), index=False)
        ck.to_csv(os.path.join(a.csv, a.tag + "_checkpoints.csv"), index=False)
    print("wrote figures to", a.out)


if __name__ == "__main__":
    main()