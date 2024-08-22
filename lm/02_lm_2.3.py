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

'''
1.为什么二阶导数比一阶导数的开销更大？
    - 二阶导数的计算，需要先求出一阶导数，然后再求出一阶导数的导数，因此二阶导数的计算开销更大。
2.在执行反向传播函数之后，立即再次执行它，看看会发生什么？
    - 会报错，因为在执行反向传播函数之后，计算图中的梯度信息已经被清空，再次执行反向传播函数时，无法计算梯度。
    - 默认情况下，计算图在反向传播后会被释放
    - 如果你确实需要多次调用 backward()，可以在第一次调用时通过将 retain_graph=True 作为参数传递给 backward() 函数来保留计算图。这会使计算图在第一次调用后仍然保留，以便再次使用。.backward(retain_graph=True)
    - 或者使用使用 zero_grad() 清除梯度
'''

a = torch.arange(4, dtype=torch.float32,requires_grad=True)
b = a.T.dot(a)
b.backward(retain_graph=True)
print(a.grad)
b.backward()
print(a.grad)
# x.grad.zero_()

'''
设计一个求控制流梯度的例子
'''

import torch
x = torch.tensor(-1.0, requires_grad=True)
def sim_control_flow(x):
    if x>0:
        y = x** 2
    else:
        y = -x ** 3
    return y

y = sim_control_flow(x)

y.backward()

print(f"输入 x:{x.item()}")
print(f"输出 y:{y.item()}")
print(f"y对x的梯度:{x.grad.item()}")

'''
使f(x)=sin(x),绘制f(x)和f'(x)的图像，其中后者不使用f'(x)=cos(x)
'''

import numpy as np

def f(x):
    return np.sin(x)

def grad_f(x):
    h = 0.0001
    return (f(x+h) - f(x-h)) / (2*h)

import matplotlib.pyplot as plt

x = np.arange(0, 10, 0.1)
plt.plot(x, f(x), label='f(x)')
plt.plot(x, grad_f(x), label="f'(x)")

plt.legend()
plt.show()

