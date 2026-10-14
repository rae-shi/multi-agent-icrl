#!/usr/bin/env bash
# Port the code we build on from the OFFICIAL ICRL-benchmarks-public repo into algos/icrl/.
#
# Usage (from anywhere):
#   bash scripts/port_icrl_code.sh                 # pinned commit below
#   COMMIT=<hash> bash scripts/port_icrl_code.sh   # a different upstream commit
#   FORCE=1 bash scripts/port_icrl_code.sh         # overwrite an existing algos/icrl
#
# Steps:
#   1. clone the official repo from GitHub and check out the pinned commit
#   2. copy the code directories into algos/icrl/ (not data, saved models, commonroad, .git)
#   3. keep the BSD-3-Clause LICENSE and write algos/icrl/PROVENANCE.md
#
# The copied code keeps its own top-level imports (common, constraint_models, utils,
# stable_baselines3, ...), so run it with algos/icrl on PYTHONPATH.
set -euo pipefail

UPSTREAM_URL="https://github.com/Guiliang/ICRL-benchmarks-public.git"
COMMIT="${COMMIT:-0d79d2c5bdf8e5fe87f89626036a423a806c4c90}"

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DEST="$REPO/algos/icrl"

# One-time copy: refuse to overwrite an existing port so later edits are never wiped.
if [ -d "$DEST" ] && [ -n "$(ls -A "$DEST" 2>/dev/null)" ] && [ "${FORCE:-0}" != "1" ]; then
  echo "$DEST is not empty. Re-run with FORCE=1 only if you want to overwrite it." >&2
  exit 1
fi

# Work in the job's local scratch when there is one.
TMP="$(mktemp -d -p "${SLURM_TMPDIR:-${TMPDIR:-/tmp}}")"
trap 'rm -rf "$TMP"' EXIT

# Every network git step has a time limit, so a stall fails loudly instead of hanging.
git_step() {
  timeout 600 git "$@" || { echo "git $1 failed or timed out (10 min limit)" >&2; exit 1; }
}

# Download only what we copy: no history, no file contents until checkout, and a sparse
# checkout that skips data/, save_model/ and the other large folders (a few MB, not 250 MB).
echo "=== fetching $COMMIT from $UPSTREAM_URL (code only) ==="
git_step clone --filter=blob:none --no-checkout --depth 1 "$UPSTREAM_URL" "$TMP/up"
git -C "$TMP/up" sparse-checkout set --no-cone '/*' '!/data/' '!/save_model/' \
  '!/commonroad_environment/' '!/images&others/'
timeout 600 git -C "$TMP/up" fetch --depth 1 origin "$COMMIT" \
  || { echo "could not fetch commit $COMMIT from $UPSTREAM_URL (not found, or timed out after 10 min)" >&2; exit 1; }
git_step -C "$TMP/up" checkout --quiet "$COMMIT"
FULL_COMMIT="$(git -C "$TMP/up" rev-parse HEAD)"
COMMIT_DATE="$(git -C "$TMP/up" log -1 --format=%cd --date=short)"
[ -f "$TMP/up/LICENSE" ] || { echo "no LICENSE in upstream checkout; stop and check" >&2; exit 1; }

echo "=== copying code ($FULL_COMMIT) -> $DEST ==="
rm -rf "$DEST"
mkdir -p "$DEST"
# top-level exclusions are anchored (./name); the rest match at any depth
tar -C "$TMP/up" -cf - \
  --anchored \
  --exclude=./.git --exclude=./data --exclude=./save_model \
  --exclude=./commonroad_environment --exclude='./images&others' \
  --no-anchored \
  --exclude=__pycache__ --exclude='*.pyc' \
  . | tar -C "$DEST" -xf -

cat > "$DEST/PROVENANCE.md" << EOF
# Provenance

Code in this directory is copied from
https://github.com/Guiliang/ICRL-benchmarks-public (BSD 3-Clause, see \`LICENSE\`),
the code of Liu et al., "Benchmarking Constraint Inference in Inverse Reinforcement
Learning", ICLR 2023.

- Upstream commit: $FULL_COMMIT (committed $COMMIT_DATE)
- Copied on: $(date +%Y-%m-%d)
- Not copied: \`.git\`, \`data/\`, \`save_model/\`, \`commonroad_environment/\`, \`images&others/\`.

## Known issues seen with the original setup (not applied; fix only if they occur)

- numpy 1.24 removed the \`np.bool\` alias. With a newer numpy this raises an
  AttributeError in \`stable_baselines3/common/base_class.py\` (about line 348) and
  \`stable_baselines3/common/vec_env/dummy_vec_env.py\` (about line 33); change both to
  \`np.bool_\`. Our venv uses numpy 1.22.4, where the alias still works.
- \`constraint_models/constraint_net/gail_net.py\`, \`reshape_actions\`: actions with more
  than two dimensions are not flattened. Seen with the HalfCheetah GAIL run on multi-env
  data. The fix was adding \`elif len(acs.shape) > 2: acs, _ = self.flatten(acs)\` after the
  discrete-actions branch.
- \`config/mujuco_BlockedHalfCheetah/train_GAIL_HCWithPos-v0.yaml\` has
  \`device: 'cuda'\`; the earlier cluster runs used \`cpu\`. Set \`device\` explicitly in
  every new config.

Later changes made in this repository are in its git history.

## Algorithm names: paper vs code

Status: hypothesis from names, configs and \`gail_net.py\`; not yet verified in the code.

| Paper | Code name | Evidence |
|---|---|---|
| GACL | GAIL | \`GailDiscriminator.cost_function\` returns 1 - D(obs, acs); GAIL configs set \`learn_cost: true\` |
| BC2L | Binary | name match |
| MECL | ICRL | by elimination (baseline list: BC2L, GACL, ICRL, VICRL) |
| VICRL | VICRL | name match |
| PPO-Lag | PPO-Lag | name match |
| (reference) | PPO | unconstrained baseline |

## Running

\`\`\`bash
export PYTHONPATH="\$PWD/algos/icrl:\$PYTHONPATH"
\`\`\`

\`algos/icrl/stable_baselines3\` is a modified SB3 0.9.0. Because it is first on
PYTHONPATH it shadows any pip-installed stable-baselines3.
EOF

echo
echo "=== done ==="
du -sh "$DEST"
echo "upstream commit: $FULL_COMMIT"
echo "next: export PYTHONPATH=\"$DEST:\$PYTHONPATH\" and try:"
echo "  python -c 'import stable_baselines3, common, constraint_models; print(\"imports ok\")'"
