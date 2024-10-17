# %%
# 池化可视化
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

# 创建示例输入 (28, 28, 1) - 单通道
input_data = np.random.random((1, 28, 28, 1))  # 模拟一张 28x28 的灰度图像

# 创建模型1，使用 padding='valid'
model_valid = tf.keras.Sequential([
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='valid', input_shape=(28, 28, 1))
])

# 创建模型2，使用 padding='same'
model_same = tf.keras.Sequential([
    tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=2, padding='same', input_shape=(28, 28, 1))
])

# 计算 MaxPooling 的输出
output_valid = model_valid(input_data)
output_same = model_same(input_data)

# 输出形状对比
print(f"Input shape: {input_data.shape}")
print(f"Valid padding output shape: {output_valid.shape}")
print(f"Same padding output shape: {output_same.shape}")

# 可视化
plt.figure(figsize=(12, 6))

# 原始输入图像
plt.subplot(1, 3, 1)
plt.imshow(input_data[0, :, :, 0], cmap='gray')
plt.title("Original Image (28x28)")

# Valid padding 输出图像
plt.subplot(1, 3, 2)
plt.imshow(output_valid[0, :, :, 0], cmap='gray')
plt.title("Valid Padding (14x14)")

# Same padding 输出图像
plt.subplot(1, 3, 3)
plt.imshow(output_same[0, :, :, 0], cmap='gray')
plt.title("Same Padding (14x14)")

plt.show()
# %% AlexNet 网络结构识别手写数字
import tensorflow as tf
from keras import layers, models,datasets

# 加载数据集
minst = datasets.mnist
# 训练集和测试集
(train_images, train_labels), (test_images, test_labels) = minst.load_data()
# 数据预处理
train_images = train_images/255.0
test_images = test_images/255.0
# 调整图像形状
train_images = train_images.reshape(-1,28,28,1)
test_images = test_images.reshape(-1,28,28,1)
# 创建AlexNet模型
model = models.Sequential([
    # 第一层卷积
    layers.Conv2D(32, (3,3), strides=(1,1), activation
    ='relu', input_shape=(28,28,1)),
    # 第一层池化
    layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    # 第二层卷积 使用64个3x3的卷积核，步长为1，填充方式为same
    layers.Conv2D(64,(3,3),strides=(1,1)
                  ,padding='same',activation='relu'),
    layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    # 第三层卷积 使用128个3x3的卷积核，步长为1，填充方式为same
    layers.Conv2D(128,(3,3),strides=(1,1)
                  ,padding='same',activation='relu'),
    # 第四层卷积 使用128个3x3的卷积核，步长为1，填充方式为same
    layers.Conv2D(128,(3,3),strides=(1,1)
                  ,padding='same',activation='relu'),
    # 第五层卷积 使用64个3x3的卷积核，步长为1，填充方式为same
    layers.Conv2D(256,(3,3),strides=(1,1),
                  padding='same',activation='relu'),
    layers.MaxPool2D(pool_size=(2,2), strides=(2,2)),
    layers.Flatten(),
    # 第一个全连接层
    layers.Dense(512, activation='relu'),
    layers.Dense(512, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# 编译模型
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
# 训练模型
model.fit(train_images, train_labels, epochs=5, batch_size=128,validation_split=0.2)


# %%
# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test Accuracy: {test_acc}")

# %%
# 预测
predictions = model.predict(test_images)
# print(predictions[0])
print(f"Predicted label: {np.argmax(predictions[0])}, True label: {test_labels[0]}")

# %%
# 可视化预测结果
import matplotlib.pyplot as plt

# 随机选择一张测试图像
index = np.random.randint(0, len(test_images))
image = test_images[index]
label = test_labels[index]

# 预测
prediction = model.predict(np.expand_dims(image, axis=0))
predicted_label = np.argmax(prediction)

# 显示图像和预测结果
plt.figure()
plt.imshow(image[:, :, 0], cmap='gray')
plt.title(f"True label: {label}, Predicted label: {predicted_label}")
plt.axis('off')
plt.show()

# %%
