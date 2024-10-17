# %%
import tensorflow as tf
from keras import layers, models, datasets

# 加载数据集
mnist = datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# 数据预处理
train_images = train_images.reshape((60000, 28, 28, 1)).astype('float32') / 255
test_images = test_images.reshape((10000, 28, 28, 1)).astype('float32') / 255

# NiN模型
def build_nin(input_shape=(28, 28, 1), num_classes=10):
    model = models.Sequential()
    model.add(layers.Conv2D(192, kernel_size=5, padding='same', activation='relu', input_shape=input_shape))
    model.add(layers.Conv2D(160, kernel_size=1, activation='relu'))
    model.add(layers.Conv2D(96, kernel_size=1, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=3, strides=2, padding='same'))
    
    model.add(layers.Conv2D(192, kernel_size=5, padding='same', activation='relu'))
    model.add(layers.Conv2D(192, kernel_size=1, activation='relu'))
    model.add(layers.Conv2D(192, kernel_size=1, activation='relu'))
    model.add(layers.MaxPooling2D(pool_size=3, strides=2, padding='same'))
    
    model.add(layers.Conv2D(192, kernel_size=3, padding='same', activation='relu'))
    model.add(layers.Conv2D(192, kernel_size=1, activation='relu'))
    model.add(layers.Conv2D(10, kernel_size=1, activation='relu'))
    
    model.add(layers.GlobalAveragePooling2D())
    model.add(layers.Dense(num_classes, activation='softmax'))
    
    return model

# 构建和编译模型
model = build_nin()
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# 训练模型
model.fit(train_images, train_labels, epochs=5, batch_size=64, validation_split=0.1)

# 评估模型
test_loss, test_acc = model.evaluate(test_images, test_labels)
print(f"Test accuracy: {test_acc}")

# 打印模型摘要
model.summary()
# %%