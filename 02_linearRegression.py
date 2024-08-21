''' 线性回归
线性回归介绍
线性回归api初步使用
数学求导
线性回归的损失【mse】和优化【梯度下降】 
最小二乘法 theta = (X^T * X)^-1 * X^T * y
梯度下降方法介绍【批量梯度下降，随机梯度下降，小批量梯度下降，动能，Adam】,算法选择
线性回归再介绍
案例：波士顿房价预测
欠拟合和过拟合
'''

import joblib
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

# 波士顿房价多项式扩展
from sklearn.preprocessing import PolynomialFeatures
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# def PolynomialRegression(degree):
#     return Pipeline([
#         ("poly", PolynomialFeatures(degree=degree, include_bias=True)),
#         ("std_scaler", StandardScaler()),
#         ("lin_reg", LinearRegression())
#     ])

# poly_reg = PolynomialRegression(degree=2)
path = os.path.join(os.path.dirname(__file__), "model")
loaded_model = joblib.load(os.path.join(path,'poly_reg.pkl'))
poly_reg = loaded_model
poly_reg.fit(X_train, y_train)
print(poly_reg.named_steps["lin_reg"].coef_)
print(poly_reg.named_steps["lin_reg"].intercept_)
y_predict = poly_reg.predict(X_test)
print(poly_reg.score(X_train, y_train))
print(poly_reg.score(X_test, y_test))
y_train_hat = poly_reg.predict(X_train)

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

path = os.path.join(os.path.dirname(__file__), "model")
joblib.dump(poly_reg, os.path.join(path,"poly_reg.pkl"))
joblib.dump(linear, os.path.join(path,"linear.pkl"))


# #%% 
# import numpy as np

# W = np.array([[1, 2 , 3, 4]])
# print(W.shape)
# print(W.T.shape)

'''
`scikit-learn` (sklearn) 的 API 主要分为以下几类：

### 1. **监督学习 (Supervised Learning)**
   - **分类 (Classification)**: 用于离散目标变量的预测。
     - 常见模型：`LogisticRegression`, `SVC` (支持向量机), `RandomForestClassifier`, `KNeighborsClassifier` 等。
   - **回归 (Regression)**: 用于连续目标变量的预测。
     - 常见模型：`LinearRegression`, `Ridge`, `Lasso`, `RandomForestRegressor`, `SVR` (支持向量回归) 等。

### 2. **无监督学习 (Unsupervised Learning)**
   - **聚类 (Clustering)**: 将数据分为多个簇（群）。
     - 常见模型：`KMeans`, `DBSCAN`, `AgglomerativeClustering` 等。
   - **降维 (Dimensionality Reduction)**: 将高维数据转换为低维表示。
     - 常见模型：`PCA` (主成分分析), `t-SNE`, `TruncatedSVD` 等。
   - **密度估计 (Density Estimation)**: 估计数据的概率分布。
     - 常见模型：`KernelDensity`, `GaussianMixture` 等。

### 3. **模型选择 (Model Selection)**
   - **交叉验证 (Cross-validation)**: 用于评估模型性能的技术。
     - 常见函数：`train_test_split`, `cross_val_score`, `GridSearchCV`, `RandomizedSearchCV` 等。
   - **超参数调优 (Hyperparameter Tuning)**: 通过网格搜索或随机搜索来优化模型参数。
     - 常见工具：`GridSearchCV`, `RandomizedSearchCV` 等。

### 4. **预处理 (Preprocessing)**
   - **数据变换 (Data Transformation)**: 用于数据的标准化、归一化和其他形式的变换。
     - 常见工具：`StandardScaler`, `MinMaxScaler`, `Normalizer`, `PolynomialFeatures` 等。
   - **特征选择 (Feature Selection)**: 用于选择最重要的特征。
     - 常见工具：`SelectKBest`, `RFE` (递归特征消除), `VarianceThreshold` 等。

### 5. **模型评估 (Model Evaluation)**
   - **指标 (Metrics)**: 用于评估模型性能的指标。
     - 常见工具：`accuracy_score`, `precision_score`, `recall_score`, `f1_score`, `roc_auc_score`, `mean_squared_error` 等。
   - **混淆矩阵 (Confusion Matrix)**: 用于评估分类模型的表现。
     - 工具：`confusion_matrix`, `classification_report` 等。

### 6. **管道 (Pipeline)**
   - **管道 (Pipeline)**: 用于将多个步骤（如预处理、模型训练）组合成一个连贯的工作流。
     - 工具：`Pipeline`, `make_pipeline` 等。

### 7. **特征提取 (Feature Extraction)**
   - **从文本或图像中提取特征**。
     - 工具：`CountVectorizer`, `TfidfVectorizer`（用于文本）, `PCA`（用于图像等数据）等。

### 8. **异常检测 (Anomaly Detection)**
   - **检测数据中的异常点**。
     - 常见模型：`IsolationForest`, `OneClassSVM`, `EllipticEnvelope` 等。

通过这些分类，`scikit-learn` 提供了丰富的工具来构建、训练、评估和优化各种机器学习模型。

**a.** 深入了解 `Pipeline` 的使用方法和好处。  
**b.** 研究如何使用 `GridSearchCV` 进行超参数优化。
'''