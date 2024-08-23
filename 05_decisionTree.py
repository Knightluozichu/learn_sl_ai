'''决策树
决策树（Decision Tree）是一种常用的机器学习算法，既可以用于分类问题，也可以用于回归问题。它通过对数据集进行递归分割，构建一棵树状结构，从而实现决策的过程。以下是关于决策树的全面复习。

### 1. **决策树的基本概念**

- **节点（Node）**：
  - **根节点（Root Node）**：树的顶端节点，代表整个数据集。
  - **内部节点（Internal Node）**：根据特定特征进行分割的节点，内部节点包含决策条件。
  - **叶节点（Leaf Node）**：不再分裂的终端节点，代表最终的预测结果（类别标签或回归值）。

- **分裂（Splitting）**：
  - 决策树通过在节点上选择一个特征并根据该特征的某个阈值进行分裂，将数据集划分成两个或多个子集。

- **树的深度（Depth of Tree）**：
  - 决策树的深度是从根节点到叶节点的最长路径上的节点数。

- **信息增益（Information Gain）**：
  - 信息增益是用于选择分裂特征的标准之一。它基于熵（Entropy）的减少量来衡量分裂的质量。

### 2. **决策树的构建过程**

1. **选择最优特征分裂数据**：
   - 使用特定的标准（如信息增益、基尼指数、方差减少等）选择一个特征，并确定一个阈值，将数据集分裂成两个或多个子集。
   
2. **递归地构建子树**：
   - 对于每个子集，递归地选择最优特征进行分裂，直到满足停止条件。

3. **停止条件**：
   - 树的最大深度达到预设值。
   - 子集中的样本数少于某个最小阈值。
   - 所有数据点在一个节点内属于同一类别（对于分类问题）。
   - 当前节点的样本不能再被有效分裂。

4. **预测**：
   - 对于一个新的样本，从根节点开始，依据样本特征逐层走到叶节点，叶节点的值就是最终的预测结果。

### 3. **决策树的分裂标准**

#### 1. **信息增益（Information Gain）**
- **用于分类问题**。
- 信息增益是基于熵的减少来衡量分裂质量的标准。熵是数据集的不确定性度量。

**熵公式**：
\[
\text{Entropy}(S) = -\sum_{i=1}^{c} p_i \log_2(p_i)
\]
其中，\( p_i \) 是第 \( i \) 类的概率，\( c \) 是类别数。

**信息增益公式**：
\[
\text{Information Gain}(S, A) = \text{Entropy}(S) - \sum_{v \in \text{Values}(A)} \frac{|S_v|}{|S|} \times \text{Entropy}(S_v)
\]
其中，\( S \) 是数据集，\( A \) 是特征，\( S_v \) 是特征 \( A \) 取值为 \( v \) 的子集。

#### 2. **基尼指数（Gini Index）**
- **用于分类问题**。
- 基尼指数度量了数据集的不纯度，基尼指数越低，表示数据集的纯度越高。

**基尼指数公式**：
\[
\text{Gini}(S) = 1 - \sum_{i=1}^{c} p_i^2
\]
其中，\( p_i \) 是第 \( i \) 类的概率。

#### 3. **方差减少（Variance Reduction）**
- **用于回归问题**。
- 方差减少通过衡量数据集的方差减少量来选择分裂特征。

**方差减少公式**：
\[
\text{Variance}(S) = \frac{1}{n} \sum_{i=1}^{n} (y_i - \bar{y})^2
\]
其中，\( y_i \) 是第 \( i \) 个样本的目标值，\( \bar{y} \) 是目标值的均值。

### 4. **决策树的优缺点**

**优点**：
- **简单直观**：决策树易于理解和解释，决策过程可以可视化为树状结构。
- **无需特征缩放**：决策树不需要特征标准化或归一化，因为它基于特征的相对顺序进行分裂。
- **处理非线性关系**：决策树能够捕捉特征与目标变量之间的复杂非线性关系。
- **可处理多种数据类型**：决策树可以同时处理数值型和类别型数据。

**缺点**：
- **容易过拟合**：如果不加限制，决策树可能会过度拟合训练数据，导致在测试数据上的表现较差。
- **对小数据集不稳定**：小的变动可能导致决策树结构的巨大变化。
- **对噪声敏感**：决策树对数据中的噪声和异常值敏感，可能会导致过拟合。

### 5. **防止过拟合的方法**

- **剪枝（Pruning）**：通过限制树的深度或修剪不必要的分支来防止过拟合。剪枝可以分为预剪枝和后剪枝。
  
  - **预剪枝（Pre-pruning）**：在构建树的过程中提前停止分裂，限制最大深度、最小样本数等。
  - **后剪枝（Post-pruning）**：在构建完整的决策树后，回头修剪那些贡献较小或不必要的分支。

- **设置最大深度**：限制决策树的最大深度，防止其无限制地生长。

- **设置最小样本数**：在分裂前限制每个节点的最小样本数，以避免对小样本的过度拟合。

- **集成学习**：通过将决策树作为基础学习器，构建如随机森林（Random Forest）和梯度提升树（Gradient Boosting Trees）等集成模型来减少过拟合。

### 6. **决策树的应用**

- **分类问题**：如垃圾邮件分类、信用评分、疾病诊断等场景中，决策树可以用于分类任务。
  
- **回归问题**：如房价预测、股票市场预测等场景中，决策树可以用于回归任务。

- **特征选择**：决策树的特征重要性排序可以用于特征选择，帮助识别最有影响力的特征。

- **决策支持**：在医疗诊断、故障诊断等场景中，决策树可以帮助决策者做出更加透明和解释性强的决策。

### 7. **决策树的代码示例（使用Python和Scikit-learn）**

以下是使用 `scikit-learn` 实现决策树的简单示例：

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建决策树分类器
clf = DecisionTreeClassifier(criterion='gini', max_depth=3)

# 训练模型
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 评估模型
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 可视化决策树
plt.figure(figsize=(12,8))
plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
plt.show()
```

### 总结

决策树是一种直观且灵活的机器学习算法，适用于多种分类和回归任务。它能够处理复杂的非线性关系并生成易于解释的模型。然而，决策树容易过拟合，且对小数据集和噪声较为敏感，因此通常需要通过剪枝、设置最大深度、最小样本数等方法来防止过拟合。此外，决策树在集成学习方法中的应用（如随机森林、梯度提升树）也非常广泛，是提升模型性能的有效手段。
'''

from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn import metrics
import matplotlib.pyplot as plt
from sklearn.tree import plot_tree

# 加载数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# 创建决策树分类器
clf = DecisionTreeClassifier(criterion='gini', max_depth=3)

# 训练模型
clf.fit(X_train, y_train)

# 预测
y_pred = clf.predict(X_test)

# 评估模型
accuracy = metrics.accuracy_score(y_test, y_pred)
print(f'Accuracy: {accuracy}')

# 可视化决策树
# plt.figure(figsize=(12,8))
# plot_tree(clf, filled=True, feature_names=iris.feature_names, class_names=iris.target_names)
# plt.show()


# 信息熵

