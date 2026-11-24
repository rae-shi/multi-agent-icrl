"""Add an `agent` argument (default 'Ant') to the MultiGoal pair adapter so ids like
SafetyPointMultiGoal1Pair-v0 build Safety<Agent>MultiGoal<level>-v0. Run from the repo root.
Every replacement must match exactly once, otherwise nothing is written."""
import sys

PATH = "envs/wrappers/safety_multigoal_pair.py"
REPL = [
    ('''def level_from_env_id(env_id):
    """'SafetyAntMultiGoal1Pair-v0' -> 1"""
    return int(env_id.split("MultiGoal")[1].replace(PAIR_SUFFIX, ""))
''',
     '''def level_from_env_id(env_id):
    """'SafetyAntMultiGoal1Pair-v0' -> 1"""
    return int(env_id.split("MultiGoal")[1].replace(PAIR_SUFFIX, ""))


def agent_from_env_id(env_id):
    """'SafetyPointMultiGoal1Pair-v0' -> 'Point'; 'SafetyAntMultiGoal1Pair-v0' -> 'Ant'"""
    head = env_id.split("MultiGoal")[0]
    return head[len("Safety"):] if head.startswith("Safety") else head
'''),
    ("    def __init__(self, level=1, inner_env=None, fall_penalty=0.0):",
     "    def __init__(self, level=1, inner_env=None, fall_penalty=0.0, agent=\"Ant\"):"),
    ('inner_env = safety_gymnasium.make("SafetyAntMultiGoal{0}-v0".format(level))',
     'inner_env = safety_gymnasium.make("Safety{0}MultiGoal{1}-v0".format(agent, level))'),
    ("        self.level = level\n",
     "        self.level = level\n        self.agent = agent\n"),
    ("    level = level_from_env_id(env_id)\n",
     "    level = level_from_env_id(env_id)\n    agent = agent_from_env_id(env_id)\n"),
    ("SafetyMultiGoalPair(level=level, fall_penalty=fall_penalty)",
     "SafetyMultiGoalPair(level=level, fall_penalty=fall_penalty, agent=agent)"),
]

src = open(PATH).read()
for i, (old, new) in enumerate(REPL, 1):
    n = src.count(old)
    if n != 1:
        sys.exit("replacement {0} matched {1} times (expected 1); nothing written".format(i, n))
    src = src.replace(old, new)
open(PATH, "w").write(src)
print("patched", PATH, "-", len(REPL), "replacements")