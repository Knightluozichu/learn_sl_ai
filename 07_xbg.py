

'''
bagging与boosting的区别
区别一:数据方面
Bagging：对数据进行采样训练；
Boosting：根据前一轮学习结果调整数据的重要性。
区别二:投票方面
Bagging：所有学习器平权投票；
Boosting：对学习器进行加权投票。
区别三:学习顺序
Bagging的学习是并行的，每个学习器没有依赖关系；
Boosting学习是串行，学习有先后顺序。
区别四:主要作用
Bagging主要用于提高泛化性能（解决过拟合，也可以说降低方差）
Boosting主要用于提高训练精度 （解决欠拟合，也可以说降低偏差）
'''
#  %% 使用集成学习 GBDT
from sklearn.ensemble import GradientBoostingRegressor

from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
import os,sys

data_url = os.path.join(os.path.dirname(__file__), "data/housing.data")
raw_df = pd.read_csv(data_url, sep="\s+", skiprows=22, header=None)
data = np.hstack([raw_df.values[::2, :], raw_df.values[1::2, :2]])
target = raw_df.values[1::2, 2]
print(data.shape, target.shape)
X_train, X_test, y_train, y_test = train_test_split(data, target, test_size=0.2, random_state=42)
# 初始化 GBDT 回归模型
gbdt = GradientBoostingRegressor(n_estimators=1000, learning_rate=0.1, max_depth=2, random_state=42)
gbdt.fit(X_train, y_train)

# 计算训练集和测试集上的 R2 分数
train_score = gbdt.score(X_train, y_train)
test_score = gbdt.score(X_test, y_test)

print(f'Train R2 Score: {train_score:.4f}')
print(f'Test R2 Score: {test_score:.4f}')
print(f'y_test[10] predictions[10]: {y_test[10]} {gbdt.predict(X_test[10].reshape(1, -1))}')
# %%
