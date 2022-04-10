#resource: https://blog.csdn.net/weixin_38314865/article/details/104311969?spm=1001.2101.3001.6650.1&utm_medium=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_default&depth_1-utm_source=distribute.pc_relevant.none-task-blog-2%7Edefault%7ECTRLIST%7ERate-1.pc_relevant_default&utm_relevant_index=2
#resource: https://zhuanlan.zhihu.com/p/159477597
#resource: https://pytorch.org/docs/stable/generated/torch.nn.functional.one_hot.html

import torch
import torch.nn as nn

import ipdb;ipdb.set_trace()
# Example of target with class indices
loss = nn.CrossEntropyLoss()
# number of classes
N = 5
# batch size
B = 3

#input
input = torch.randn(B, N, requires_grad=True)
target = torch.empty(B, dtype=torch.long).random_(5)
one_hot = nn.functional.one_hot(target,num_classes=N)

####################NLLLOSS#######################
#reference:https://zhuanlan.zhihu.com/p/383044774
nllloss = nn.NLLLoss()
nllloss_output = -(input*one_hot).sum()/B
nllloss_torch = nllloss(input, target)

print(f'two different ways for calculating NLLLOSS: {nllloss_output}, {nllloss_torch}')
####################cross Entropy Loss########################
softmax = nn.Softmax(dim=1)
softmax_input = softmax(input)

log_softmax_input = torch.log(softmax_input)
crossEntropyLoss = -(log_softmax_input * one_hot).sum()/B

output = loss(input, target)
print(f'two different ways for calculating crossEntropyloss: {crossEntropyLoss}, {output}')
output.backward()

# Example of target with class probabilities
input = torch.randn(3, 5, requires_grad=True)
target = torch.randn(3, 5).softmax(dim=1)
output = loss(input, target)
output.backward()
