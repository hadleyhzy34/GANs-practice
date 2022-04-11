#resource: https://machinelearningmastery.com/divergence-between-probability-distributions/
#resource: https://zhuanlan.zhihu.com/p/425693597

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

# resource: https://stackoverflow.com/questions/49886369/kl-divergence-for-two-probability-distributions-in-pytorch

import ipdb;ipdb.set_trace()
##############KL-divergence###############
p = torch.Tensor([0.36,0.48,0.16])
q = torch.Tensor([0.333, 0.333, 0.333])

kl_div_1 = (p * (p/q).log()).sum()

kl_div_2 = F.kl_div(q.log(), p, None, None, 'sum')

print(f'test to see if {kl_div_1} equals to {kl_div_2}')

# test case
w = torch.Tensor([0.1,0.2,0])
kl_div_3 = (p * (p/w).log()).sum()
print(f'current output kl divergence distance is: {kl_div_3}')

b = torch.Tensor([0.1, 0.3, 0])
kl_div_4 = (w * (w/b).log()).sum()
print(f'current output kl divergence distance is: {kl_div_4}')

kl_div = lambda p,q: (p * (p/q).log()).sum()

js_div = lambda p,q: 0.5 * kl_div(p, 0.5 * (p+q)) + 0.5 * kl_div(q, 0.5 * (p+q))

js_div_1 = js_div(p,q)
js_div_2 = 0.5 * (F.kl_div((0.5*(p+q)).log(),p,None,None,'sum') + F.kl_div((0.5*(p+q)).log(),q,None,None,'sum'))
print(f'current value is: {js_div_1}, {js_div_2}')
