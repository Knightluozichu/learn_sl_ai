from sklearn.linear_model import LinearRegression

x = [
    [80,86],
    [82,80],
    [85,78],
    [90,90],
    [86,82],
    [82,90],
    [78,80],
    [92,94]
]

y = [84.2,80.6,80.1,90,83.2,87.6,79.4,93.4]

estimator = LinearRegression()
estimator.fit(x, y)

print(estimator.coef_)

print(estimator.intercept_)

print(estimator.predict([[100,80]]))

# 损失函数 cost function 线性回归（预测）：均方误差，最小二乘法 逻辑回归（分类）：交叉熵损失函数

import numpy as np

# 样本数据
X = np.array([[1, 1], [1, 2], [2, 2], [2, 3]])
y  = np.dot(X, np.array([1, 2])) + 3

# 添加偏置项
X_b = np.c_[np.ones((4, 1)), X]
print(X_b)

# 最小二乘法
theta_best = np.linalg.inv(X_b.T.dot(X_b)).dot(X_b.T).dot(y)
print(theta_best)

# 预测
X_new = np.array([[1, 3]])
X_new_b = np.c_[np.ones((1, 1)), X_new]
y_predict = X_new_b.dot(theta_best)
print(y_predict)

# 使用sklearn 梯度下降求解theta
from sklearn.linear_model import LinearRegression
lin_reg = LinearRegression()
lin_reg.fit(X_b, y)
print(lin_reg.intercept_, lin_reg.coef_)
y_predict = lin_reg.predict(X_new_b)
print(y_predict)

# 波士顿房价预测

import pandas as pd
import numpy as np
import os,sys

data_url = os.path.join(os.path.dirname(__file__), "data/housing.data")
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
print(data.shape, target.shape)

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
linear = LinearRegression()
linear.fit(X_train, y_train)
print(linear.coef_)
print(linear.intercept_)
y_predict = linear.predict(X_test)
print(linear.score(X_train, y_train))
print(linear.score(X_test, y_test))
y_train_hat = linear.predict(X_train)

import matplotlib.pyplot as plt
# 训练集
plt.figure(num="train")
plt.plot(range(len(X_train)), y_train, 'r', label='u true')
plt.plot(range(len(X_train)), y_train_hat, 'g', label='u predict')
plt.legend(loc="upper right")
plt.show()

# 测试集
plt.figure(num="test")
plt.plot(range(len(X_test)), y_test, 'r', label='u true')
plt.plot(range(len(X_test)), y_predict, 'g', label='u predict')
plt.legend(loc="upper right")
plt.show()
