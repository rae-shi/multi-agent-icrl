import numpy as np, safety_gymnasium as sg
e = sg.make('SafetyAntMultiGoal1-v0')
a0, a1 = 'agent_0', 'agent_1'
s0, s1 = e.observation_space(a0), e.observation_space(a1)
print('obs bounds identical:', np.array_equal(s0.low, s1.low) and np.array_equal(s0.high, s1.high))
print('action spaces identical:', e.action_space(a0) == e.action_space(a1))
pat = [(float(l), float(h)) for l, h in zip(s0.low, s0.high)]
blocks, start = [], 0
for i in range(1, len(pat) + 1):
    if i == len(pat) or pat[i] != pat[start]:
        blocks.append((start, i, pat[start])); start = i
print('blocks (start, end, (low, high)):', blocks)
e.reset(seed=0)
n_same = 0
for t in range(50):
    out = e.step({a: e.action_space(a).sample() for a in (a0, a1)})
    if t == 0: print('step() returns', len(out), 'items')
    n_same += np.allclose(out[0][a0], out[0][a1])
print('steps with identical obs for both agents:', n_same, '/ 50')