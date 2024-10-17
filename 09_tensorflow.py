#%%
import tensorflow as tf
# %%
import numpy as np

# %%
# 创建int32类型的0维张量，即标量
rank_0_tensor = tf.constant(4)
print(rank_0_tensor)
# %%
rank_1_tensor = tf.constant([2.0,3.0,4.0])
print(rank_1_tensor)
rank_2_tensor=tf.constant([[1,2],[3,4],[5,6]])
print(rank_2_tensor)
# %%
rank_3_tensor=tf.constant([
                           [[0,1,2,3,4],[5,6,7,8,9]],
                           [[10,11,12,13,14],[15,16,17,18,19]],
                           [[20,21,22,23,24],[25,26,27,28,29]]
                           ])
print(rank_3_tensor)
# %%
# 转成numpy
np_rank_3_tensor = np.array(rank_3_tensor)
# np_rank_3_tensor = rank_3_tensor.numpy() 
print(np_rank_3_tensor)
print( type(np_rank_3_tensor) )
print(np_rank_3_tensor.shape)
# %% 数学运算
a = tf.constant([[1,2],[3,4]])
b = tf.constant([[5,6],[7,8]])
print(tf.add(a,b)) # 加法
print(tf.multiply(a,b)) # 对应元素相乘
print(tf.matmul(a,b)) # 矩阵相乘


# %%
# 求和
print(tf.reduce_sum(a))
# 求平均
print(tf.reduce_mean(a))
# 求最大值
print(tf.reduce_max(a))
# 求最小值
print(tf.reduce_min(a))
# 求最大值的索引
print(tf.argmax(a))
# 求最小值的索引
print(tf.argmin(a))
# %%
# 变量
# 创建变量
a = tf.Variable(1.0)
print(a)
# 变量是一种特殊的张量，形状和类型不可改变，但值可以改变
# 修改变量的值
a.assign(2.0)
print(a)

# %%
# 。流程：数据获取、数据预处理、模型构建、模型训练、模型评估、模型预测
# 1. 数据获取
# 使用cnn_mnist数据集
mnist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()
# 数据预处理
# 归一化
train_images, test_images = train_images / 255.0, test_images / 255.0
# 添加一个维度
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]
# 转换为tensor
train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)
train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int32)
test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)
test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int32)
# 创建数据集
train_dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))
train_dataset = train_dataset.shuffle(60000).batch(64)
test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.batch(64)
# 2. 模型构建
'''
tf.keras.layers.Conv2D是 TensorFlow 中用于二维卷积操作的层，其主要参数及含义如下：
filters参数
含义：整数，输出空间的维度（即卷积核的数量）。它决定了卷积层输出特征图的数量。
示例：如果设置filters=32，则该卷积层将对输入进行卷积操作，生成 32 个不同的特征图。
kernel_size参数
含义：一个整数或由两个整数组成的元组 / 列表，指定卷积核的大小。例如，(3, 3)表示使用 3x3 的卷积核。
示例：较小的卷积核（如 3x3）在计算机视觉任务中非常常见，因为它们可以有效地捕捉局部特征，同时减少参数数量和计算量。较大的卷积核（如 5x5 或 7x7）可以捕捉更广泛的上下文信息，但会增加计算成本。
strides参数
含义：一个整数或由两个整数组成的元组 / 列表，指定卷积核在输入上移动的步长。默认值为 (1, 1)。
示例：如果设置strides=(2, 2)，则卷积核在水平和垂直方向上每次移动两个像素。较大的步长可以减少输出特征图的尺寸，但可能会丢失一些细节信息。较小的步长可以保留更多的信息，但会增加计算量和输出特征图的尺寸。
padding参数
含义：'valid'或'same'。'valid'表示不进行填充，卷积操作只在输入的有效区域进行，可能会导致输出特征图的尺寸小于输入。'same'表示进行填充，使得输出特征图的尺寸与输入相同。
示例：在图像分类任务中，通常使用'same'填充以保持特征图的尺寸稳定，特别是在网络的早期层。而在一些需要减少计算量或特定输出尺寸的情况下，可以使用'valid'填充。
data_format参数
含义：字符串，指定输入数据的格式。可以是'channels_last'（默认值，表示输入形状为(batch_size, height, width, channels)）或'channels_first'（表示输入形状为(batch_size, channels, height, width)）。
示例：如果输入数据是图像，并且图像的通道维度在最后一个位置（如 RGB 图像的形状为(batch_size, height, width, 3)），则使用默认的'channels_last'。如果输入数据的通道维度在第一个位置（如某些深度学习框架的默认格式），则需要设置为'channels_first'。
dilation_rate参数
含义：一个整数或由两个整数组成的元组 / 列表，指定膨胀卷积的膨胀率。默认值为 (1, 1)。
示例：膨胀卷积可以在不增加参数数量的情况下扩大卷积核的感受野。例如，设置dilation_rate=(2, 2)，则卷积核的每个元素之间会有一个像素的间隔，从而使感受野扩大。
activation参数
含义：激活函数，可以是一个可调用对象或函数名称。如果不指定，默认为线性激活（即不进行激活）。
示例：常见的激活函数有'relu'（修正线性单元）、'sigmoid'、'tanh'等。例如，设置activation='relu'可以在卷积层后应用 ReLU 激活函数，增加模型的非线性表达能力。
use_bias参数
含义：布尔值，指定是否在卷积层中使用偏置项。默认值为 True。
示例：使用偏置项可以为每个输出特征图添加一个可学习的偏移量，有助于模型更好地拟合数据。但在某些情况下，为了减少模型的参数数量或防止过拟合，可以设置use_bias=False。
kernel_initializer参数
含义：卷积核的初始化方法。可以是一个可调用对象或字符串，表示预定义的初始化方法。
示例：常见的初始化方法有'glorot_uniform'（Xavier 均匀初始化）、'he_normal'等。例如，设置kernel_initializer='he_normal'可以使用 He 正态分布初始化卷积核，有助于在深度神经网络中更好地传播梯度。
bias_initializer参数
含义：偏置项的初始化方法。与kernel_initializer类似，可以是一个可调用对象或字符串，表示预定义的初始化方法。
示例：默认情况下，偏置项通常初始化为零。但可以根据需要选择其他初始化方法，如'zeros'（初始化为零）、'ones'（初始化为一）等。
kernel_regularizer参数
含义：卷积核的正则化方法。可以是一个正则化函数或正则化对象。
示例：常见的正则化方法有 L1 正则化和 L2 正则化，可以通过设置kernel_regularizer=tf.keras.regularizers.l1_l2(l1=0.01, l2=0.01)来同时应用 L1 和 L2 正则化，防止过拟合。
bias_regularizer参数
含义：偏置项的正则化方法。与kernel_regularizer类似，可以是一个正则化函数或正则化对象。
示例：如果需要对偏置项进行正则化，可以设置bias_regularizer=tf.keras.regularizers.l2(0.01)来应用 L2 正则化。
activity_regularizer参数
含义：层的输出的正则化方法。可以是一个正则化函数或正则化对象。
示例：例如，可以设置activity_regularizer=tf.keras.regularizers.l2(0.01)来对卷积层的输出进行 L2 正则化。
name参数
含义：字符串，为该层指定一个名称。这在构建复杂模型时有助于区分不同的层。
示例：可以设置name='conv_layer_1'来为卷积层命名，方便在调试和分析模型时进行识别。
'''
model = tf.keras.models.Sequential([
    tf.keras.layers.Conv2D(32, 3, activation='relu', input_shape=(28, 28, 1)),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Conv2D(64, 3, activation='relu'),
    tf.keras.layers.MaxPooling2D(),
    tf.keras.layers.Flatten(),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(10, activation='softmax')
])

'''
model.compile()
没有特定的返回值用于常规使用：这个函数用于配置模型的训练参数，如优化器、损失函数和评估指标等。它不返回一个可直接用于后续操作的值，但它确保模型在训练前进行了正确的设置。
model.fit()
返回值：它返回一个History对象。
作用：这个对象包含了训练过程中的历史信息，如每一轮的损失值和评估指标的值等。可以通过这个对象来分析模型的训练过程和性能。例如，可以使用history.history['loss']来获取训练过程中的损失值列表，history.history['accuracy']（如果设置了准确率评估指标）来获取准确率的变化情况等。这对于监控模型的训练进度、判断是否过拟合以及调整超参数非常有帮助。
model.evaluate()
返回值：它返回一个包含损失值和评估指标值的列表。
作用：这个列表通常包含两个或多个值，第一个值是损失值，后面的值是设置的评估指标的值。这个返回值可以用于评估模型在测试集或验证集上的性能表现。例如，可以将这个值与训练过程中的损失和指标值进行比较，以判断模型是否过拟合或欠拟合，以及确定模型在实际应用中的效果。
'''
# 3. 模型训练

model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
'''
model.fit()是 TensorFlow 和 Keras 中用于训练模型的函数，其主要参数及含义如下：
x和y参数
含义：输入数据和对应的目标数据。可以是 numpy 数组、TensorFlow 的张量或生成器等形式。
示例：如果是图像分类问题，x可以是一批图像数据，y可以是对应的类别标签。
batch_size参数
含义：在训练过程中每次迭代使用的样本数量。
示例：如果设置为 32，那么模型在每次迭代中会处理 32 个样本，然后更新权重。较小的批次大小可能会使训练过程更加随机，但计算效率较低；较大的批次大小可以提高计算效率，但可能会导致模型收敛到局部最优解。
epochs参数
含义：模型训练的轮数。每一轮表示模型遍历整个训练数据集一次。
示例：如果设置为 10，模型将对训练数据进行 10 次完整的遍历。增加训练轮数可能会使模型更好地拟合数据，但也可能导致过拟合。
validation_data参数
含义：用于在训练过程中进行验证的数据。可以是一个由验证输入数据和验证目标数据组成的元组。
示例：(x_val, y_val)，其中x_val是验证集的输入数据，y_val是对应的目标数据。在每一轮训练结束后，模型会在验证集上进行评估，以监控模型的性能和防止过拟合。
callbacks参数
含义：一个回调函数的列表。回调函数可以在训练过程中的不同阶段执行特定的操作，如早停法（EarlyStopping）、保存模型权重（ModelCheckpoint）等。
verbose参数
含义：控制训练过程中的输出信息的详细程度。
取值及含义：
0：安静模式，不输出任何信息。
1：进度条显示每个轮次的训练信息。
2：每个轮次输出一行信息，显示损失和评估指标。
shuffle参数
含义：在训练过程中是否对训练数据进行随机打乱。
取值及含义：
True：在每个轮次开始前随机打乱训练数据，有助于提高模型的泛化能力。
False：不打乱数据，按照数据的原始顺序进行训练。通常情况下，打乱数据是一个好的做法，但在某些特定情况下，如处理时间序列数据时，可能需要按照顺序进行训练。
'''
model.fit(train_dataset, epochs=5)
# 4. 模型评估
'''
model.compile()是 TensorFlow 和 Keras 中用于配置模型训练参数的函数，其主要参数及含义如下：
optimizer参数
含义：优化器，用于更新模型的权重以最小化损失函数。它决定了模型在训练过程中如何根据损失函数的梯度来调整参数。
常见优化器及特点：
'sgd'（随机梯度下降）：简单但可能需要较多的训练时间和合适的学习率调整。可以通过设置学习率、动量等参数进行优化。例如，SGD(learning_rate=0.01, momentum=0.9)。
'adam'（自适应矩估计）：通常在大多数情况下表现良好，能够自动调整学习率，收敛速度较快。Adam()通常使用默认参数就能取得较好的效果，但也可以根据需要调整学习率等参数。
'rmsprop'（均方根传播）：也是一种自适应学习率的优化器，对梯度的缩放具有较好的稳定性。RMSprop(learning_rate=0.001)是一个常见的设置。
loss参数
含义：损失函数，用于衡量模型的预测值与真实值之间的差异。根据不同的任务类型选择不同的损失函数。
常见损失函数及适用场景：
对于二分类问题，可以使用'binary_crossentropy'（二元交叉熵）。例如，在判断图像是猫还是狗的二分类任务中，模型的输出是一个介于 0 和 1 之间的概率值，表示属于某一类的可能性，二元交叉熵损失函数可以衡量这个预测概率与真实标签之间的差异。
对于多分类问题，可以使用'categorical_crossentropy'（分类交叉熵）。当模型的输出是多个类别的概率分布时，分类交叉熵损失函数可以有效地衡量预测的概率分布与真实的类别分布之间的差异。例如，在识别手写数字的任务中，模型输出是 10 个类别的概率分布，对应 0 到 9 十个数字。
对于回归问题，可以使用'mean_squared_error'（均方误差）或'mean_absolute_error'（平均绝对误差）。均方误差是预测值与真实值之差的平方的平均值，对较大的误差比较敏感；平均绝对误差是预测值与真实值之差的绝对值的平均值，对异常值相对不那么敏感。
metrics参数
含义：评估指标列表，用于在训练和评估过程中监控模型的性能。可以是预定义的指标名称字符串，也可以是自定义的评估函数。
常见评估指标及意义：
'accuracy'（准确率）：在分类问题中，准确率是正确分类的样本数与总样本数之比。它是衡量分类模型性能的一个直观指标。例如，在识别动物种类的任务中，如果有 100 张图像，模型正确分类了 80 张，那么准确率就是 80%。
'precision'（精确率）、'recall'（召回率）和'f1_score'（F1 分数）：主要用于二分类问题的更细致的评估。精确率是真正例（预测为正且实际为正）与预测为正的样本数之比；召回率是真正例与实际为正的样本数之比；F1 分数是精确率和召回率的调和平均数。例如，在疾病检测任务中，精确率高表示模型很少将健康人误判为患者，召回率高表示模型能够尽可能多地检测出真正的患者。
'mae'（平均绝对误差）和'mse'（均方误差）：在回归问题中，平均绝对误差是预测值与真实值之差的绝对值的平均值，均方误差是预测值与真实值之差的平方的平均值。它们可以衡量回归模型的预测值与真实值之间的接近程度。例如，在预测房价的任务中，平均绝对误差或均方误差越小，说明模型的预测越准确。
loss_weights参数（可选）
含义：当模型有多个输出并且每个输出都有一个损失函数时，可以使用这个参数为不同的损失函数分配权重。
示例：如果模型有两个输出，一个是分类任务，另一个是回归任务，并且希望给予分类任务的损失函数两倍的权重，可以设置loss_weights=[2, 1]。
sample_weight_mode参数（可选）
含义：指定样本权重的模式。可以是 "temporal"（用于时间序列数据）或 "None"（默认值，表示不使用样本权重）等。
示例：在处理时间序列数据时，如果不同时间点的样本重要性不同，可以设置样本权重模式为 "temporal"，并通过fit函数的sample_weight参数传入相应的样本权重数组。
weighted_metrics参数（可选）
含义：当使用样本权重时，这个参数可以指定要加权的评估指标。与metrics参数类似，它可以是预定义的指标名称字符串或自定义的评估函数。
示例：如果在分类任务中使用样本权重，并且希望对准确率进行加权评估，可以将'accuracy'添加到weighted_metrics参数中。
'''
evalution =  model.evaluate(test_dataset)
print(evalution)

# %%
# 5. 模型预测
pred = model.predict(test_images)
# print(pred)
# 可视化结果
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].numpy().reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(np.argmax(pred[i]))
plt.show()

# %%
# 使用skleran和tensorflow划分鸢尾花数据集
from sklearn.model_selection import train_test_split
import tensorflow as tf
import numpy as np
from sklearn import datasets
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
# 加载数据
iris = datasets.load_iris()
# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
# 根据数据特征可视化数据
iris_d = pd.DataFrame(iris['data'], columns = ['Sepal_Length',
                                               'Sepal_Width',
                                               'Petal_Length',
                                               'Petal_Width'])
iris_d['Species'] = iris.target

# 使用pairplot可视化所有特征两两相关的子图
sns.pairplot(iris_d, hue='Species', markers=["o", "s", "D"])
plt.suptitle('Iris species shown by color', y=1.02)  # 调整标题位置
plt.show()
#  %% sklearn处理鸢尾花数据集
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, classification_report
from sklearn.preprocessing import StandardScaler
# 划分数据集
x_train, x_test, y_train, y_test = train_test_split(iris.data, iris.target, test_size=0.2)
# 数据标准化
scaler = StandardScaler()
x_train = scaler.fit_transform(x_train)
x_test = scaler.transform(x_test)
# 创建模型
knn = KNeighborsClassifier(n_neighbors=3)
# 训练模型
knn.fit(x_train, y_train)
# 预测
y_pred = knn.predict(x_test)
# 评估
print(accuracy_score(y_test, y_pred))
print(classification_report(y_test, y_pred))
# # 可视化结果
# plt.figure(figsize=(10,10))
# for i in range(25):
#     plt.subplot(5,5,i+1)
#     plt.xticks([])
#     plt.yticks([])
#     plt.grid(False)
#     plt.imshow(x_test[i].reshape(2, 2), cmap=plt.cm.binary)
#     plt.xlabel(y_pred[i])
# plt.show()

# %%

import tensorflow as tf
import numpy as np
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split

# 加载鸢尾花数据集
iris = load_iris()
X = iris.data
y = iris.target

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)
# 转换标签为 one-hot 编码
# 进行独热编码
def one_hot_encode_object_array(arr):
    # 去重获取全部的类别
    uniques, ids = np.unique(arr, return_inverse=True)
    # 返回热编码的结果
    return tf.keras.utils.to_categorical(ids, len(uniques))

y_train_one_hot = one_hot_encode_object_array(y_train)
y_test_one_hot = one_hot_encode_object_array(y_test)

# 创建模型
model = tf.keras.Sequential([
    tf.keras.layers.Dense(10, activation='relu', input_shape=(4,)),
    tf.keras.layers.Dense(10, activation='relu'),
    tf.keras.layers.Dense(3, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# 训练模型
history = model.fit(X_train, y_train_one_hot, epochs=10, batch_size=1, validation_split=0.1)

# 评估模型
loss_fn, accuracy = model.evaluate(X_test, y_test_one_hot, verbose=0)
# print(f"Test loss: {loss}, Test accuracy: {accuracy}")

# 进行预测
predictions = model.predict(X_test)
print('Accuracy score:', accuracy_score(y_test_one_hot.argmax(axis=1), predictions.argmax(axis=1)))
print(classification_report(y_test_one_hot.argmax(axis=1), predictions.argmax(axis=1)))

# model.summary()

# 获取模型训练过程的准确率以及损失率的变化
accuracy = history.history['accuracy']
loss_fn = history.history['loss']
val_loss = history.history['val_loss']
val_accuracy = history.history['val_accuracy']
epochs = range(len(accuracy))
plt.plot(epochs, accuracy, 'b', label='Training accuracy')
plt.plot(epochs, val_accuracy, 'orange', label='Validation accuracy')
plt.title('Training and validation accuracy')
plt.legend()
plt.show()

plt.plot(epochs, loss_fn, 'b', label='Training Loss')
plt.plot(epochs, val_loss, 'orange', label='Validation Loss')
plt.title('Training and validation loss')
plt.legend()
plt.show()




# %% 激活函数
# 创建X数据
X = np.linspace(-10, 10, 100)
# 创建激活函数
y = tf.nn.sigmoid(X)
# 创建绘图
plt.plot(X, y)
plt.title('Sigmoid Function')
plt.show()
# %%
# 创建激活函数
y = tf.nn.relu(X)
# 创建绘图
plt.plot(X, y)
plt.title('Relu Function')
plt.show()
# %%
# 创建激活函数
y = tf.nn.tanh(X)
# 创建绘图
plt.plot(X, y)
plt.title('Tanh Function')
# %%
# 创建激活函数
y = tf.nn.softmax(X)
# 创建绘图
plt.plot(X, y)
plt.title('Softmax Function')
plt.show()
# %%
# 创建激活函数
y = tf.nn.leaky_relu(X)
# 创建绘图
plt.plot(X, y)
plt.title('Leaky Relu Function')
plt.show()
# %%
# 创建激活函数
y = tf.nn.elu(X)
# 创建绘图
plt.plot(X, y)
plt.title('Elu Function')
plt.show()
# %%
# 创建激活函数
y = tf.nn.softplus(X)
# 创建绘图
plt.plot(X, y)
plt.title('Softplus Function')
plt.show()
# %%
# 创建激活函数
y = tf.nn.softsign(X)
# 创建绘图
plt.plot(X, y)
plt.title('Softsign Function')
plt.show()
# %%
# 创建激活函数
y = tf.nn.relu6(X)
# 创建绘图
plt.plot(X, y)
plt.title('Relu6 Function')
plt.show()
# %%
# 创建激活函数
y = tf.nn.selu(X)
# 创建绘图
plt.plot(X, y)
plt.title('Selu Function')
plt.show()

# %%
# 创建激活函数
y = tf.nn.gelu(X)
# 创建绘图
plt.plot(X, y)
plt.title('Gelu Function')
plt.show()
# %%
# 创建激活函数
y = tf.nn.swish(X)
# 创建绘图
plt.plot(X, y)
plt.title('Swish Function')
plt.show()
# %%
# 创建激活函数
y = tf.nn.silu(X)
# 创建绘图
plt.plot(X, y)
plt.title('Silu Function')
plt.show()

# %% 损失函数
# 创建X数据
X = np.linspace(-10, 10, 100)
# 创建f(x) = x^2
y = X ** 2 + np.random.randn(100) * 10
y_pred = X**2
# # 创建绘图
# plt.plot(X, y)
# plt.title('Square Loss Function')
# plt.show()

loss_fn =  tf.keras.losses.MeanSquaredError()

print(loss_fn(y, y_pred).numpy())

# 绘图
plt.plot(X, y, label='True')
plt.plot(X, y_pred, label='Predicted')
plt.title('Square Loss Function')
plt.legend()
plt.show()

# %% LeNet-5
import tensorflow as tf
import numpy as np

minist = tf.keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = minist.load_data()
# 数据预处理
train_images, test_images = train_images / 255.0, test_images / 255.0
train_images = train_images[..., tf.newaxis]
test_images = test_images[..., tf.newaxis]
# 转换为tensor
train_images = tf.convert_to_tensor(train_images, dtype=tf.float32)
train_labels = tf.convert_to_tensor(train_labels, dtype=tf.int32)
test_images = tf.convert_to_tensor(test_images, dtype=tf.float32)
test_labels = tf.convert_to_tensor(test_labels, dtype=tf.int32)
# 创建数据集
dataset = tf.data.Dataset.from_tensor_slices((train_images, train_labels))

dataset_size = len(list(dataset))
# 划分训练集和验证集
train_size = int(0.8 * dataset_size)
val_size = dataset_size - train_size

train_dataset = dataset.take(train_size)
val_dataset = dataset.skip(train_size)

train_dataset = train_dataset.shuffle(train_size).batch(64)
val_dataset = val_dataset.batch(64)

test_dataset = tf.data.Dataset.from_tensor_slices((test_images, test_labels))
test_dataset = test_dataset.batch(64)

# 创建模型
def build_dynamic_cnn(input_shape):
    # 创建一个模型
    model = tf.keras.Sequential()
    
    # 动态添加卷积层，根据输入形状调整层数
    if input_shape[1] >= 64:
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                         activation='relu', padding='same',
                                         input_shape=input_shape))
        model.add(tf.keras.layers.MaxPooling2D())
    
    # 添加第二层卷积层，无论输入大小如何
    model.add(tf.keras.layers.Conv2D(filters=32, kernel_size=(3, 3),
                                     activation='relu', padding='same'))
    model.add(tf.keras.layers.MaxPooling2D())

    # 动态调整全连接层，根据输入的通道数
    # 如果输入是灰度图
    if input_shape[2] == 1:
        model.add(tf.keras.layers.Conv2D(filters=16, kernel_size=(3, 3),
                                         activation='relu', padding='same'))
    # 如果输入是彩色图
    else:
        model.add(tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3),
                                         activation='relu', padding='same'))
    # 展平层，准备进入全连接层
    model.add(tf.keras.layers.Flatten())

    # 添加全连接层
    model.add(tf.keras.layers.Dense(128, activation='relu'))

    # 输出层，输出10个类别
    model.add(tf.keras.layers.Dense(10, activation='softmax'))

    # 编译模型
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

input_shape = train_images.shape[1:]
model = build_dynamic_cnn(input_shape)
model.summary()
# 训练模型
model.fit(train_dataset, validation_data=val_dataset, epochs=5)

# 评估模型
model.evaluate(test_dataset)

# %%
model.summary()
# model.save('dynamic_cnn.h5')

# %%
# 预测模型
pred = model.predict(test_images)
# 可视化结果
import matplotlib.pyplot as plt
plt.figure(figsize=(10,10))
for i in range(25):
    plt.subplot(5,5,i+1)
    plt.xticks([])
    plt.yticks([])
    plt.grid(False)
    plt.imshow(test_images[i].numpy().reshape(28, 28), cmap=plt.cm.binary)
    plt.xlabel(np.argmax(pred[i]))
plt.show()

# %%
