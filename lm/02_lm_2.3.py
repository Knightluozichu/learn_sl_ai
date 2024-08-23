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
绘制函数y=f(x)=x^3-1/x的图像，同时绘制其在x=1处的导数切线
'''

import numpy as np
import matplotlib.pyplot as plt

x = np.linspace(-10, 10, 100)
y = x**3 - 1/x
plt.plot(x, y, label='y=x^3-1/x')
f_y = 3*x*x + 1/x**2
# 绘制x=1时，y=x^3-1/x的导数切线
x1 = 1
plt.plot(x1, 3*1*1 + 1/1**2)
plt.scatter(x1, 3*1*1 + 1/1**2, color='red')
# plt.legend()
# plt.show()

'''
求函数f(x)=3*x1^2 + 5*e^x2的梯度
'''

import torch

x = torch.tensor([1.0, 2.0], requires_grad=True)
y = 3*x[0]**2 + 5*torch.exp(x[1])
y.backward()
print(f'f(x)=3*x1^2 + 5*e^x2的梯度:{x.grad}')

'''
函数f(x)=||x||_2的梯度是啥？
'''

x = torch.arange(4, dtype=torch.float32, requires_grad=True)
y = torch.linalg.norm(x)
y.backward()
print(f'f(x)=||x||_2的值：{y} ,梯度:{x.grad}')

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
b = torch.dot(x,x)
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

# plt.legend()
# plt.show()


'''
概率
'''