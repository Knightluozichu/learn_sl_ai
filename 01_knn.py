''' KNN算法
K近邻算法介绍
knn api初步使用
kd树
案例1：鸢尾花分类
特征工程-特征与处理
k-近邻算法总结
交叉验证，网格搜索
案例2：预测facebook签到位置
'''

from sklearn.datasets import load_iris

iris = load_iris()
X = iris.data
y = iris.target
print(X.shape)
print(y.shape)

print(X[:5])
print(y[:5])

# y的种类
print(set(y))
print(iris.target_names)
print(iris.feature_names)
# print(iris.DESCR)

#
# load和fetch
# data: 数据
# target: 标签
# feature_names: 特征名称
# target_names: 标签名称
# DESCR: 数据集描述
# filename: 文件名
#


import seaborn as sns 
import pandas as pd
import matplotlib.pyplot as plt

iris_d = pd.DataFrame(iris['data'], columns = ['Sepal_Length',
                                               'Sepal_Width',
                                               'Petal_Length',
                                               'Petal_Width'])
iris_d['Species'] = iris.target

def plot_iris(iris, col1, col2):
    sns.lmplot(x= col1, y=col2, data=iris, hue='Species', fit_reg=True)
    plt.xlabel(col1) 
    plt.ylabel(col2)
    plt.title('Iris species shown by color')
    plt.show()

# plot_iris(iris_d, 'Sepal_Length', 'Sepal_Width')
# plot_iris(iris_d, 'Sepal_Length', 'Petal_Length')
plot_iris(iris_d, 'Sepal_Length', 'Petal_Width')
# plot_iris(iris_d, 'Petal_Length', 'Petal_Width')
# plot_iris(iris_d, 'Petal_Length', 'Petal_Length')
# plot_iris(iris_d, 'Petal_Length', 'Sepal_Width')

from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X_train, y_train)
y_pred = knn.predict(X_test)
print(accuracy_score(y_test, y_pred))
# 输出分类报告
print(classification_report(y_test, y_pred, target_names=iris.target_names))
