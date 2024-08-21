from matplotlib import axis
import torch

X = torch.arange(2*3*4,dtype=torch.float32).reshape(2, 3, 4)
# print(X)
# print(len(X))
# print(X.shape)
# print(X.size())
# print(X.numel())
# A = X / X.sum(axis=1)
# print(A)
# print(4000 * 0.30)
sum_1 = X.sum(axis=0, keepdim=True)
print(sum_1.shape)
sum_1 = X.sum(axis=1, keepdim=True)
print(sum_1.shape)
sum_1 = X.sum(axis=2, keepdim=True)
print(sum_1.shape)
lig_1 = torch.linalg.norm(X)
print(lig_1)
Y = torch.tensor([3.0,-4.0])
print(torch.linalg.norm(Y))

from torch.linalg import norm
print(norm(X))