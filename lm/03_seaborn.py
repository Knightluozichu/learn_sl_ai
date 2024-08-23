'''
当然可以！下面我会通过几个具体的案例，带你逐步入门 Seaborn，帮助你掌握这个强大的数据可视化库的基本用法。

### 1. **案例1：绘制基本的分布图**

**目标**：了解如何使用 Seaborn 绘制单变量的分布图，探索数据的分布情况。

**数据集**：使用 Seaborn 自带的 `tips` 数据集，这个数据集记录了餐厅的小费数据。

```python
import seaborn as sns
import matplotlib.pyplot as plt

# 加载数据集
tips = sns.load_dataset("tips")

# 绘制单变量分布图（直方图和核密度图）
sns.histplot(tips['total_bill'], kde=True)

# 显示图表
plt.show()
```

**解释**：
- `sns.histplot()`：用于绘制直方图，并使用 `kde=True` 选项来叠加核密度估计（KDE）曲线，展示数据的分布。

### 2. **案例2：绘制散点图，分析两个变量的关系**

**目标**：学习如何使用散点图来展示两个变量之间的关系。

```python
# 绘制散点图
sns.scatterplot(data=tips, x="total_bill", y="tip", hue="day", style="time", size="size")

# 显示图表
plt.show()
```

**解释**：
- `sns.scatterplot()`：用于绘制散点图。通过 `hue` 参数根据不同的天数用不同的颜色区分数据点，`style` 参数根据时间区分点的形状，`size` 参数用点的大小来表示聚餐人数。

### 3. **案例3：绘制箱线图，探索数据的分布情况**

**目标**：使用箱线图来查看不同类别数据的分布情况，并识别潜在的异常值。

```python
# 绘制箱线图
sns.boxplot(x="day", y="total_bill", data=tips, palette="Set2")

# 显示图表
plt.show()
```

**解释**：
- `sns.boxplot()`：用于绘制箱线图。箱线图展示了数据的五数概括（最小值、第一四分位数、中位数、第三四分位数、最大值）以及异常值。

### 4. **案例4：绘制热力图，分析变量之间的相关性**

**目标**：学习如何使用热力图展示变量之间的相关性矩阵。

```python
# 计算相关性矩阵
corr = tips.corr()

# 绘制热力图
sns.heatmap(corr, annot=True, cmap="coolwarm")

# 显示图表
plt.show()
```

**解释**：
- `sns.heatmap()`：用于绘制热力图。`annot=True` 表示在图上标注相关系数值，`cmap="coolwarm"` 指定热力图的配色方案。

### 5. **案例5：绘制成对关系图，探索多变量之间的关系**

**目标**：使用成对关系图（pair plot）查看多个变量之间的两两关系。

```python
# 绘制成对关系图
sns.pairplot(tips, hue="day", palette="Set1")

# 显示图表
plt.show()
```

**解释**：
- `sns.pairplot()`：用于绘制成对关系图，展示数据集中每对变量之间的关系，并根据 `hue` 参数用不同颜色表示不同类别的数据。

### 6. **案例6：绘制分类散点图，展示分类变量和连续变量的关系**

**目标**：使用分类散点图展示分类变量和连续变量之间的关系。

```python
# 绘制分类散点图
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)

# 显示图表
plt.show()
```

**解释**：
- `sns.stripplot()`：用于绘制分类散点图，`jitter=True` 添加随机抖动，防止点的重叠，以便更好地观察分布情况。

### 7. **案例7：自定义图表样式和主题**

**目标**：学习如何自定义 Seaborn 图表的样式和主题，使图表更加美观和专业。

```python
# 设置主题
sns.set_theme(style="whitegrid")

# 绘制箱线图
sns.boxplot(x="day", y="total_bill", data=tips, palette="Set2")

# 添加标题和标签
plt.title("Box Plot of Total Bill by Day")
plt.xlabel("Day of Week")
plt.ylabel("Total Bill ($)")

# 显示图表
plt.show()
```

**解释**：
- `sns.set_theme(style="whitegrid")`：设置 Seaborn 的全局样式，这里使用了白色背景带网格的样式。
- 使用 `plt.title()`、`plt.xlabel()` 和 `plt.ylabel()` 添加标题和轴标签，使图表更加清晰和易于理解。

### 总结

通过这些案例，你可以逐步掌握 Seaborn 的基础功能。Seaborn 的强大之处在于它能够简化许多复杂的可视化任务，并且其与 Pandas 的良好集成使得数据分析更加高效。随着你对 Seaborn 的熟悉程度提高，你可以尝试创建更加复杂和自定义的图表，用于探索和展示数据。
'''

import seaborn as sns
import matplotlib.pyplot as plt

# 餐厅小费数据集
# 加载数据集
tips = sns.load_dataset("tips")
print("tips数据集：")
print(tips)
print(f"预览： {tips.head()}")
print(f"数据集信息： {tips.info()}")
print(f"数据集描述统计： {tips.describe()}")
print(f"数据集缺失值统计： {tips.isnull().sum()}")
print(f"数据集唯一值统计： {tips.nunique()}")
print(f"数据集类型： {tips.dtypes}")
print(f"数据集列名： {tips.columns}")
print(f"数据集形状： {tips.shape}")
print(f"数据集大小： {tips.size}")
print(f"数据集维度： {tips.ndim}")

# 绘制单变量分布图（直方图和核密度图）
sns.histplot(tips['total_bill'], kde=True)

# 显示图表
plt.show()

# 绘制散点图，分析两个变量的关系
	# •	sns.scatterplot()：用于绘制散点图。通过 hue 参数根据不同的天数用不同的颜色区分数据点，style 参数根据时间区分点的形状，size 参数用点的大小来表示聚餐人数。
sns.scatterplot(data=tips, x="total_bill", y ="tip", hue="day", style="time", size="size")
plt.show()

# 案例3：绘制箱线图，探索数据的分布情况
# 绘制箱线图
sns.boxplot(x="day", y="total_bill", data=tips, palette="Set2")
plt.show()
'''
这个图表是一个箱线图（Box Plot），它展示了餐厅在不同的日子里（周四到周日）总账单金额（`total_bill`）的分布情况。以下是对这个箱线图的解读：

### 1. **中位数（Median）**
- 每个箱子的中间线表示总账单金额的中位数。
- 可以看到，不同日子的中位数有所不同：周四和周五的中位数较低，而周六和周日的中位数相对较高。

### 2. **四分位数（Quartiles）**
- 箱子的上下边缘分别表示第一四分位数（Q1，25%分位）和第三四分位数（Q3，75%分位）。
- 周六的箱子较高，表明这天的账单金额分布范围更广，波动较大；而周五的箱子较短，账单金额分布更集中。

### 3. **胡须（Whiskers）**
- 从箱子延伸出来的线（胡须）表示非异常值的范围，通常是从第一四分位数减去1.5倍四分位距（IQR）到第三四分位数加上1.5倍四分位距的范围。
- 胡须的长度不同，表明在不同的日子里，账单金额的分布范围也不同。

### 4. **异常值（Outliers）**
- 在胡须之外的点被称为异常值，它们表示远离大多数账单金额的少数值。
- 可以看到，周四、周五、周六和周日都有一些异常值，特别是在周六和周日有几笔高额账单（超过40的账单）。

### 5. **整体趋势**
- 从这个图表可以看出，总体来说，周末（周六和周日）的账单金额比工作日（周四和周五）更高，且分布更广，这可能反映了更多人选择在周末外出就餐，或者点的菜更贵。

### 总结：
- **周四和周五**：账单金额较低且集中，较少有高额账单。
- **周六和周日**：账单金额较高且分布更广，出现了更多的高额账单和异常值。

这个图表有助于我们了解在不同的日子里，餐厅的账单金额是如何变化的，哪些日子账单金额较高，以及在哪些日子更容易出现高额账单。
'''

#  案例4：绘制热力图，分析变量之间的相关性
# 删除非数值型列
tips_numeric = tips.select_dtypes(include=[float, int])

# 计算相关性矩阵
corr = tips_numeric.corr()

# 绘制热力图
sns.heatmap(corr, annot=True, cmap="coolwarm")

# 显示图表
plt.show()

# 案例5：绘制成对关系图，探索多变量之间的关系
# 绘制成对关系图
sns.pairplot(tips, hue="day", palette="Set1")
plt.show()

# 案例6：绘制分类散点图，展示分类变量和连续变量的关系
# 绘制分类散点图
sns.stripplot(x="day", y="total_bill", data=tips, jitter=True)
plt.show()

# 案例7：自定义图表样式和主题
# 设置主题
sns.set_theme(style="whitegrid")

# 绘制箱线图
sns.boxplot(x="day", y="total_bill", data=tips, order=None, hue="sex",hue_order=None,palette=None)

# 添加标题和标签
plt.title("Box Plot of Total Bill by Day")
plt.xlabel("Day of Week")
plt.ylabel("Total Bill ($)")

# 显示图表
plt.show()