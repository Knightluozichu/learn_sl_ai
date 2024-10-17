
# %%
import tensorflow as tf
from keras import layers, models, datasets
import numpy as np

# 加载数据集
minst = datasets.mnist
# 训练集和测试集
(train_images,train_labels),(test_images,test_labels) = minst.load_data()
# 数据预处理
train_images = train_images / 255.0
test_images = test_images / 255.0

# 将灰度图像转换为 RGB（3 通道）
train_images = np.stack([train_images]*3, axis=-1)
test_images = np.stack([test_images]*3, axis=-1)

# # 调整图像尺寸到 224x224
# train_images = tf.image.resize(train_images, [224, 224])
# test_images = tf.image.resize(test_images, [224, 224])

# GoogLeNet
class Inception(models.Model):
    def __init__(self, c1, c2, c3, c4):
        super(Inception, self).__init__()
        # 分支1
        self.p1_1 = layers.Conv2D(c1,(1,1),padding='same',activation='relu')
        # 分支2
        self.p2_1 = layers.Conv2D(c2[0],(1,1),padding='same',activation=
        'relu')
        self.p2_2 = layers.Conv2D(c2[1],(3,3),padding='same',activation=
        'relu')
        # 分支3
        self.p3_1 = layers.Conv2D(c3[0],(1,1),padding='same',activation=
        'relu')
        self.p3_2 =  layers.Conv2D(c3[1],kernel_size=(5,5),padding='same',activation='relu')
        # 分支4
        self.p4_1 = layers.MaxPool2D(3,1,padding='same')
        self.p4_2 = layers.Conv2D(c4,1,padding='same',activation='relu')
    
    def call(self, x):
        p1 = self.p1_1(x)
        p2 = self.p2_2(self.p2_1(x))
        p3 = self.p3_2(self.p3_1(x))
        p4 = self.p4_2(self.p4_1(x))
        return layers.concatenate([p1,p2,p3,p4])

class GoogLeNet(models.Model):
    def __init__(self):
        super(GoogLeNet, self).__init__()
        model = models.Sequential()
        # block1
        model.add(layers.Conv2D(64,7,padding='same',activation='relu',input_shape=(28,28,3)))
        model.add(layers.MaxPool2D(3,2,padding='same'))
        # block2
        model.add(layers.Conv2D(192,3,padding='same',activation='relu'))
        model.add(layers.MaxPool2D(3,2,padding='same'))
        # block3
        model.add(Inception(64,(96,128),(16,32),32))
        model.add(Inception(128,(128,192),(32,96),64))
        model.add(layers.MaxPool2D(3,2,padding='same'))
        # block4
        model.add(Inception(192,(96,208),(16,48),64))
        model.add(Inception(160,(112,224),(24,64),64))
        model.add(Inception(128,(128,256),(24,64),64))
        model.add(Inception(112,(144,288),(32,64),64))
        model.add(Inception(256,(160,320),(32,128),128))
        model.add(layers.MaxPool2D(3,2,padding='same'))
        # block5
        model.add(Inception(256,(160,320),(32,128),128))
        model.add(Inception(384,(192,384),(48,128),128))
        model.add(layers.GlobalAveragePooling2D())
        model.add(layers.Dropout(0.4))
        model.add(layers.Dense(10,activation='softmax'))
        self.model = model

    def call(self, x):
        return self.model(x)


model = GoogLeNet()
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

model.fit(train_images,train_labels,epochs=1,batch_size=64,validation_split=0.1,verbose=1,shuffle=True)
model.evaluate(test_images,test_labels)

# %%
model.summary()
# %%
