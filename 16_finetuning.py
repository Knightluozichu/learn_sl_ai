# %%
import tensorflow as tf
from keras._tf_keras.keras.preprocessing.image import ImageDataGenerator
from keras import models, layers, optimizers
import matplotlib.pyplot as plt
import os

# 数据目录
traindir = "data/hotdog-nothotdog/train"
testdir = "data/hotdog-nothotdog/test"

train_dir = os.path.join(os.path.dirname(__file__), traindir)
test_dir = os.path.join(os.path.dirname(__file__), testdir)

# 检查数据目录是否存在
if not os.path.exists(train_dir) or not os.path.exists(test_dir):
    raise FileNotFoundError("训练或测试数据目录不存在，请检查路径。")

for gpu in tf.config.experimental.list_physical_devices("GPU"):
    tf.config.experimental.set_memory_growth(gpu, True)

# %%
image_generator = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0 / 255,
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    fill_mode='nearest'
)

BATCH_SIZE = 32
IMG_HEIGHT = 224
IMG_WIDTH = 224

train_data_gen = image_generator.flow_from_directory(
    directory=str(train_dir),
    batch_size=BATCH_SIZE,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True,
    classes=["hotdog", "nothotdog"],
)

test_data_gen = image_generator.flow_from_directory(
    directory=str(test_dir),
    batch_size=BATCH_SIZE,
    target_size=(IMG_HEIGHT, IMG_WIDTH),
    shuffle=True,
    classes=["hotdog", "nothotdog"],
)

# 检查是否找到图像
if train_data_gen.samples == 0 or test_data_gen.samples == 0:
    raise ValueError("未找到任何图像，请检查数据目录和分类名称。")

def show_batch(image_batch, label_batch):
    plt.figure(figsize=(10, 10))
    for n in range(15):
        ax = plt.subplot(5, 5, n + 1)
        plt.imshow(image_batch[n])
        plt.axis("off")

image_batch, label_batch = next(train_data_gen)
# show_batch(image_batch, label_batch)

ResNet50 = tf.keras.applications.resnet_v2.ResNet50V2(
    weights="imagenet", input_shape=(224, 224, 3), include_top=False
)
for layer in ResNet50.layers:
    layer.trainable = False

net = tf.keras.models.Sequential()
net.add(ResNet50)
net.add(tf.keras.layers.GlobalAveragePooling2D())
net.add(tf.keras.layers.Dense(512, activation='relu'))
net.add(tf.keras.layers.Dropout(0.5))
net.add(tf.keras.layers.Dense(2, activation="softmax"))

net.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
            loss="categorical_crossentropy", 
            metrics=["accuracy"])

history = net.fit(
    train_data_gen,
    steps_per_epoch=train_data_gen.samples // BATCH_SIZE,
    epochs=10,
    validation_data=test_data_gen,
    validation_steps=test_data_gen.samples // BATCH_SIZE,
)