# inspect_step.py
import numpy as np, safety_gymnasium as sg
e = sg.make('SafetyAntMultiGoal1-v0')
print('reset ->', type(e.reset(seed=0)))
out = e.step({a: e.action_space(a).sample() for a in ('agent_0', 'agent_1')})
for i, x in enumerate(out):
    print(i, type(x).__name__, (list(x.keys()) if isinstance(x, dict) else x))
    if isinstance(x, dict):
        for k, v in x.items():
            if isinstance(v, dict): print('    ', k, '->', v)
            elif hasattr(v, 'shape'): print('    ', k, '-> array', v.shape)
            else: print('    ', k, '->', v)
print('contact_other_cost =', e.unwrapped.task.contact_other_cost)
EOF_NOTE = None