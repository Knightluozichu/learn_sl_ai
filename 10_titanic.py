
# %%
# if '__file__' in globals():
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
'''
PassengerId: 乘客ID, 唯一标识每位乘客的整数。
Pclass: 船票等级, 代表乘客舱位的类别。有三个等级:1 = 一等舱, 2 = 二等舱, 3 = 三等舱。
Name: 乘客姓名。
Sex: 性别, male表示男性, female表示女性。
Age: 年龄。如果年龄是小数, 表示它是估计的。对于年龄小于1岁的婴儿, 小数表示真实的年龄。某些年龄未知的乘客则为空。
SibSp: 同船的兄弟姐妹和配偶数量。兄弟姐妹定义包括继兄弟姐妹和同父异母的兄弟姐妹。配偶定义不包括未婚伴侣。
Parch: 同船的父母与子女数量。这里的某些儿童只有保姆陪同, 因此Parch=0。
Ticket: 船票号码。
Fare: 船票价格, 表示乘客为旅行票支付的费用。
Cabin: 客舱号码。某些乘客的客舱号码未知。
Embarked: 登船港口, 表示乘客登船的地点。C = Cherbourg, Q = Queenstown, S = Southampton。
'''
import os
from re import X

from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import SelectFromModel
print(__file__)
print(os.path.dirname(__file__))
parent_dir = os.path.dirname(__file__)

# titanic数据文件路径
train_titanic_file_path_pair = 'data/titanic/train.csv'
test_titanic_file_path_pair = 'data/titanic/test.csv'
train_file_path = os.path.join(parent_dir, train_titanic_file_path_pair)
test_file_path = os.path.join(parent_dir, test_titanic_file_path_pair)
import pandas as pd

# 读取数据
train_df = pd.read_csv(train_file_path)
test_df = pd.read_csv(test_file_path)

# 查看数据
print('训练数据前5行:')
print(train_df.head())

print('数据集基本信息:')
print(train_df.info())

print('数据集描述性统计信息:')
print(train_df.describe())

print('训练数据集缺失值:')
print(train_df.isnull().sum())

print('测试数据缺失值:')
print(test_df.isnull().sum())

# 绘制年龄分布直方图
import matplotlib.pyplot as plt
train_df['Age'].hist(bins=50)
plt.title('Histogram fo Age')
plt.xlabel('Age')
plt.ylabel('Frequency')
plt.show()

# 绘制年龄的箱线图
plt.boxplot(train_df['Age'].dropna())
plt.title('Boxplot of Age')
plt.xlabel('Age')
plt.ylabel('Value')
plt.show()

# 中位数填充年龄
train_df.fillna({'Age':train_df['Age'].median()}, inplace=True)
test_df.fillna({'Age':test_df['Age'].median()}, inplace=True)

# 填充登船港口
train_df['Embarked'] = train_df['Embarked'].fillna('S')

# 填充甲板号 因为数据大量缺失，不会去预测或则估计，也不能删除，因为其他的字段是有用的，所以填充为U
train_df.fillna({'Cabin':'U'}, inplace=True)

# 提取Cabin的甲板号，将Nan视为'U'
train_df['Deck'] = train_df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'U')
# print(train_data_df['Deck'].value_counts())

# 填充船票
test_df.fillna({'Fare':test_df['Fare'].median()}, inplace=True)

# print('训练数据集缺失值:')
# print(train_data_df.isnull().sum())

# print('测试数据缺失值:')
# print(test_data_df.isnull().sum())


# %%
# 转换类别变量
# 性别
train_df['Sex'] = train_df['Sex'].map({'male':0,'female':1})
test_df['Sex']  = test_df['Sex'].map({'male':0,'female':1})

# 使用one-hot登船港口
train_df = pd.get_dummies(train_df, columns=['Embarked'])
test_df = pd.get_dummies(test_df, columns=['Embarked'])

# %%
# 构造新特征
# 从Name中提取称谓作为新特征Title
train_df['Title'] = train_df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
test_df['Title'] = test_df['Name'].apply(lambda name: name.split(',')[1].split('.')[0].strip())
 
# 创建FamilySize特征
train_df['FamilySize'] = train_df['SibSp'] + train_df['Parch'] + 1
test_df['FamilySize'] = test_df['SibSp'] + test_df['Parch'] + 1
 
# 创建IsAlone特征
# 使用 loc 进行直接赋值，避免链式赋值问题
train_df.loc[train_df['FamilySize'] > 1, 'IsAlone'] = 0
train_df.loc[train_df['FamilySize'] == 1, 'IsAlone'] = 1

test_df.loc[test_df['FamilySize'] > 1, 'IsAlone'] = 0
test_df.loc[test_df['FamilySize'] == 1, 'IsAlone'] = 1
# 从Cabin特征提取甲板信息作为新特征Deck
# 如果Cabin值缺失，则使用 'U' 表示Unknown
train_df['Deck'] = train_df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'U')
test_df['Deck'] = test_df['Cabin'].apply(lambda x: x[0] if pd.notna(x) else 'U')
 
# 从Ticket特征提取票据前缀
train_df['TicketPrefix'] = train_df['Ticket'].apply(lambda x: x.split()[0] if not x.isdigit() else 'None')
test_df['TicketPrefix'] = test_df['Ticket'].apply(lambda x: x.split()[0] if not x.isdigit() else 'None')
 
# 查看数据集中新构造的特征
print(train_df[['Title', 'FamilySize', 'IsAlone', 'Deck', 'TicketPrefix']].head())
# %%
# 特征选择
from sklearn.preprocessing import LabelEncoder
# 创建 LabelEncoder 实例
label_encoder = LabelEncoder()
train_df['TicketPrefix'] = label_encoder.fit_transform(train_df['TicketPrefix'])
test_df['TicketPrefix'] = label_encoder.fit_transform(test_df['TicketPrefix'])
train_df['Deck'] = label_encoder.fit_transform(train_df['Deck'])
test_df['Deck'] = label_encoder.fit_transform(test_df['Deck'])
train_df['Title'] = label_encoder.fit_transform(train_df['Title'])
test_df['Title'] = label_encoder.fit_transform(test_df['Title'])


y_train = train_df['Survived']
X_train = train_df.drop(['Survived','Name','Ticket','Cabin','PassengerId'], axis=1)
 
forest = RandomForestClassifier(n_estimators=100, random_state=42)
forest.fit(X_train.values, y_train)
 
selector = SelectFromModel(forest, threshold='mean', prefit=True)
X_important_train = selector.transform(X_train.values)
important_feature_names = X_train.columns[selector.get_support()]
 
print("Selected features after feature selection:", important_feature_names)

# %%
# 删除无关特征
X_train = train_df[important_feature_names]
X_test = test_df[important_feature_names]



# %%
# 建立模型
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

# 分割数据进行本地验证
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.2, random_state=42)

# 归一化数据
from sklearn.preprocessing import StandardScaler
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_val = scaler.transform(X_val)
X_test = scaler.transform(X_test)

# 模型初始化
rf = RandomForestClassifier(n_estimators=100, random_state=42)

# 网格搜索
param_grid = {
    'n_estimators': [100, 200, 300, 400, 500],
    'max_depth': [5, 10, 15, 20, 25, 30]
}

grid_search = GridSearchCV(estimator=rf, param_grid=param_grid, cv=5)
grid_search.fit(X_train, y_train)
print('Best parameters:', grid_search.best_params_)
print('Best score:', grid_search.best_score_)

# %%
# 模型训练
best_clf = grid_search.best_estimator_
# 模型预测
y_pred = best_clf.predict(X_val)
# 模型评估
accuracy = accuracy_score(y_val, y_pred)
print('Accuracy:', accuracy)
print('Classification Report:\n', classification_report(y_val, y_pred))
# %%
# 组合不同模型
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import GradientBoostingClassifier

# 初始化模型
clf1 = LogisticRegression(random_state=42)
clf2 = RandomForestClassifier(random_state=42)
clf3 = SVC(random_state=42)
clf4 = DecisionTreeClassifier(random_state=42)
clf5 = GradientBoostingClassifier(random_state=42)

# 组合模型
voting_clf = VotingClassifier(estimators=[
    ('lr', clf1),
    ('rf', clf2),
    ('svc', clf3),
    ('dt', clf4),
    ('gb', clf5)
], voting='hard')

# 模型训练
voting_clf.fit(X_train, y_train)
# 模型预测
y_pred = voting_clf.predict(X_val)
# 模型评估
accuracy = accuracy_score(y_val, y_pred)
print('Accuracy:', accuracy)
print('Classification Report:\n', classification_report(y_val, y_pred))
# %%
# 模型预测
X_test = test_df[important_feature_names]
# y_pred = best_clf.predict(X_test)
y_pred = voting_clf.predict(X_test)

# 保存预测结果
test_df['Survived'] = y_pred
test_df[['PassengerId', 'Survived']].to_csv('data/titanic/submission.csv', index=False)
# %%
