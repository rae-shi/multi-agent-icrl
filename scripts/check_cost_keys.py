import safety_gymnasium as sg
e = sg.make('SafetyAntMultiGoal1-v0'); e.unwrapped.task.contact_other_cost = 0.0
e.reset(seed=0); hits = bad = 0; A = ('agent_0', 'agent_1')
for t in range(1000):
    o, r, c, te, tr, info = e.step({a: e.action_space(a).sample() for a in A})
    for a in A:
        hits += c[a] > 0
        bad += abs(float(c[a]) - info[a][a]['cost_sum']) > 1e-9
        assert 'cost_contact_other' not in info[a][a], 'contact cost still on'
    if any(te.values()) or any(tr.values()): e.reset()
print('steps with cost>0:', hits, '| cost vs cost_sum mismatches:', bad)
