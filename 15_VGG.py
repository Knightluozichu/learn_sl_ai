# %%
import tensorflow as tf
import numpy as np
from keras import layers, models, datasets

# 加载数据集
minst = datasets.mnist
# 训练集和测试集
(train_images,train_labels),(test_images,test_labels) = minst.load_data()
# 数据预处理
train_images = train_images / 255.0
test_images = test_images / 255.0

train_images = np.stack([train_images]*3, axis=-1)
test_images = np.stack([test_images]*3, axis=-1)

# 调整图像大小
train_images = tf.image.resize(train_images, (32, 32))
test_images = tf.image.resize(test_images, (32, 32))

# VGG16
def VGG16(input_shape=(32,32,3), num_classes=10):
    inputs = layers.Input(shape=input_shape)
    # block1
    x = layers.Conv2D(8,3,1,'same',activation='relu')(inputs)
    x = layers.Conv2D(8,3,1,'same',activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    # block2
    x = layers.Conv2D(16,3,1,'same',activation='relu')(x)
    x = layers.Conv2D(16,3,1,'same',activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    # block3
    x = layers.Conv2D(32,3,1,'same',activation='relu')(x)
    x = layers.Conv2D(32,3,1,'same',activation='relu')(x)
    x = layers.Conv2D(32,3,1,'same',activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    # block4
    x = layers.Conv2D(32,3,1,'same',activation='relu')(x)
    x = layers.Conv2D(32,3,1,'same',activation='relu')(x)
    x = layers.Conv2D(32,3,1,'same',activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    # block5
    x = layers.Conv2D(32,3,1,'same',activation='relu')(x)
    x = layers.Conv2D(32,3,1,'same',activation='relu')(x)
    x = layers.Conv2D(32,3,1,'same',activation='relu')(x)
    x = layers.MaxPooling2D(2,2)(x)
    # block6
    x = layers.Flatten()(x)
    x = layers.Dense(128,activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    x = layers.Dense(128,activation='relu')(x)
    # x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes,activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs)
    return model

model = VGG16()
optimizer = tf.keras.optimizers.Adam(learning_rate=0.0003)
model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
model.fit(train_images, train_labels, epochs=5)
model.evaluate(test_images, test_labels)
model.summary()
# %%