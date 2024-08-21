''' 逻辑回归
激活函数：sigmoid函数 1/(1+e^(-z)) z = w1*x1 + w2*x2 + ... + wn*xn + b， z = w^T*x + b。 Tanh函数，Relu函数
损失函数：似然对数，交叉熵损失函数
'''

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.preprocessing import StandardScaler
import numpy as np

url = "https://archive.ics.uci.edu/ml/machine-learning-databases/breast-cancer-wisconsin/breast-cancer-wisconsin.data"

# 数据列名
columns = [
    "id", "clump_thickness", "uniformity_of_cell_size", "uniformity_of_cell_shape",
    "marginal_adhesion", "single_epithelial_cell_size", "bare_nuclei", "bland_chromatin",
    "normal_nucleoli", "mitoses", "class"
]

df = pd.read_csv(url, header=None, names=columns)
# print(df.head())

# 检查 所有数据是否存在异常值"?"
# print(df.isin(['?']).sum())

# 将 'bare_nuclei' 列中的 '?' 替换为 NaN
df['bare_nuclei'].replace('?', np.nan, inplace=True)

# 将 'bare_nuclei' 列转换为数值类型
df['bare_nuclei'] = pd.to_numeric(df['bare_nuclei'])

# 去除包含 NaN 的行
df.dropna(inplace=True)

# 删除 ID 列，因为它对分类无意义
df.drop(columns=['id'], inplace=True)

# 特征矩阵和目标向量
X = df.drop(columns=['class'])
y = df['class']

# 将目标变量二值化，2 -> 0（良性），4 -> 1（恶性）
y = y.map({2: 0, 4: 1})

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 标准化特征
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# 选择Logistic Regression模型
model = LogisticRegression(random_state=42)
model.fit(X_train, y_train)

# 模型预测
y_train_pred = model.predict(X_train)
y_test_pred = model.predict(X_test)

# 评估模型
print("Train Accuracy:", accuracy_score(y_train, y_train_pred))
print("Test Accuracy:", accuracy_score(y_test, y_test_pred))

# 打印分类报告
print("\nClassification Report:\n", classification_report(y_test, y_test_pred))

# 打印混淆矩阵
print("Confusion Matrix:\n", confusion_matrix(y_test, y_test_pred))