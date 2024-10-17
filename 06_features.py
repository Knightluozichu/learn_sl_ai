'''
如果我们对特征工程(feature engineering)做一个定义，那它指的是：利用领域知识和现有数据，创造出新的特征，用于机器学习算法；可以手动(manual)或自动(automated)。

特征：数据中抽取出来的对结果预测有用的信息。
特征工程：使用专业背景知识和技巧处理数据，使得特征能在机器学习算法上发挥更好的作用的过程。
在业界有一个很流行的说法：

数据与特征工程决定了模型的上限，改进算法只不过是逼近这个上限而已。
'''
#%%
import pandas as pd
import numpy as np
import seaborn as sns

df_titanic = sns.load_dataset('titanic')

# 我们先对数据集的缺失值情况做一个了解(汇总分布)：
df_titanic.isnull().sum()

# %%在我们当前Titanic的案例中，embark_town字段有 2 个空值，考虑删除缺失处理下。

df_titanic[df_titanic["embark_town"].isnull()]
# df_titanic.dropna(axis=0,how="any",subset=['embark_town'],inplace=True)

'''
df_titanic.method({'embark_town': 'unknown'}, inplace=True)
# 或者
df_titanic['embark_town'] = df_titanic['embark_town'].fillna('unknown')
'''
#将空值作为一种特殊的属性值来处理，它不同于其他的任何属性值。如所有的空值都用unknown填充。一般作为临时填充或中间过程。
df_titanic['embark_town'] = df_titanic['embark_town'].fillna('unknown')

# %%df_titanic.isnull().sum()
fare_median = df_titanic['fare'].median()  # 计算中位数
df_titanic.fillna({'fare': fare_median}, inplace=True)  # 使用字典指定列和值，并且操作原DataFrame

# %%众数填充——embarked：只有两个缺失值，使用众数填充
print(df_titanic['embarked'].isnull().sum())
df_titanic['embarked'] =df_titanic['embarked'].fillna(df_titanic['embarked'].mode()) 
print(df_titanic['embarked'].value_counts())

# %%同类均值填充
print(df_titanic['age'].isnull().sum())
series = df_titanic.groupby(['sex', 'pclass','who'])['age'].mean()
print(series)
age_group_mean = df_titanic.groupby(['sex', 'pclass', 'who'])['age'].mean().reset_index()
print(age_group_mean)

def select_group_age_median(row):
    condition = ((row['sex'] == age_group_mean['sex']) &
                (row['pclass'] == age_group_mean['pclass']) &
                (row['who'] == age_group_mean['who']))
    return age_group_mean[condition]['age'].values[0]
df_titanic['age'] =df_titanic.apply(lambda x: select_group_age_median(x) if np.isnan(x['age']) else x['age'],axis=1)
print(df_titanic['age'].isnull().sum())
# %% 模型预测填充
# 检查 'age' 列中缺失值的数量
print(df_titanic['age'].isnull().sum())

# 选择与年龄预测相关的特征列
df_titanic_age = df_titanic[['age', 'pclass', 'sex', 'who', 'fare', 'parch', 'sibsp']]

# 将分类变量转换为独热编码（One-Hot Encoding）
df_titanic_age = pd.get_dummies(df_titanic_age)

# 显示转换后的前几行数据
# 乘客分成已知年龄和未知年龄两部分
known_age = df_titanic_age[df_titanic_age.age.notnull()]
unknown_age = df_titanic_age[df_titanic_age.age.isnull()]
# y 即目标年龄
y_for_age = known_age['age']
# X 即特征属性值
X_train_for_age = known_age.drop(['age'], axis=1)
X_test_for_age = unknown_age.drop(['age'], axis=1)
from sklearn.ensemble import RandomForestRegressor
rfr = RandomForestRegressor(random_state=0, n_estimators=2000, n_jobs=-1)
rfr.fit(X_train_for_age, y_for_age)
# 用得到的模型进行未知年龄结果预测
y_pred_age = rfr.predict(X_test_for_age)
# 用得到的预测结果填补原缺失数据
df_titanic.loc[df_titanic.age.isnull(), 'age'] = y_pred_age
sns.histplot(df_titanic['age'], kde=True)

# %%线性插值法
df_titanic['fare'].interpolate(method = 'linear', axis = 0)

#%% 检查异常值
# sns.catplot(y="fare",x="survived", kind="box", data=df_titanic,palette="Set2")
sns.scatterplot(x="fare", y="age", hue="survived",data=df_titanic,palette="Set1")
# %%
