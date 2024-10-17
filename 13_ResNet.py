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

def resnet_block(inputs,filters, kernel_size=3,stride=1,use_batch_norm=True,conv_shortcut=False):
    """
    实现一个 ResNet 残差块

    参数：
    - inputs: 输入张量
    - filters: 卷积核数量（通道数）
    - kernel_size: 卷积核大小
    - stride: 步幅
    - use_batch_norm: 是否使用批归一化
    - conv_shortcut: 是否对快捷连接进行卷积调整

    返回：
    - 输出张量
    """
    x = inputs

    # Shortcut 分支
    if conv_shortcut:
        shortcut = layers.Conv2D(filters,1,strides=stride,
                                 padding='same')(inputs)
        if use_batch_norm:
            shortcut = layers.BatchNormalization()(shortcut)
    else:
        shortcut = inputs

    # 主分支
    x = layers.Conv2D(filters,kernel_size,strides=stride,
                      padding='same')(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    x=layers.ReLU()(x)

    x=layers.Conv2D(filters,kernel_size,1,
                    padding='same')(x)
    if use_batch_norm:
        x = layers.BatchNormalization()(x)
    
    # 合并快捷连接
    x = layers.add([shortcut,x])
    output = layers.ReLU()(x)

    return output

# ResNet 模型
def build_resnet(input_shape, num_classes):
    inputs = layers.Input(shape=input_shape)

    # 初始化卷积层
    x = layers.Conv2D(64,7,strides=2,
                      padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPool2D(3,2,'same')(x)

    # 堆叠残差块 1
    num_filters = 64
    x = resnet_block(x,num_filters,conv_shortcut=False)
    x = resnet_block(x,num_filters,conv_shortcut=False)

    
    # 堆叠残差块 2
    x = resnet_block(x, 128,stride=2,conv_shortcut=True)
    x = resnet_block(x, 128,conv_shortcut=False)

    # 堆叠残差块 3
    x = resnet_block(x, 256,stride=2,conv_shortcut=True)
    x = resnet_block(x, 256,conv_shortcut=False)

    # 堆叠残差块 4
    x = resnet_block(x, 512,stride=2,conv_shortcut=True)
    x = resnet_block(x, 512,conv_shortcut=False)

    # 添加全局平均池化层
    x = layers.GlobalAveragePooling2D()(x)
    outputs = layers.Dense(num_classes,activation='softmax')(x)

    model = models.Model(inputs=inputs,outputs=outputs)
    return model

# 取train_images的最后三个shape
input_shape = train_images.shape[1:]
# 类别数
num_classes = 10
model = build_resnet(input_shape,num_classes)
model.compile(optimizer='adam',
              loss='sparse_categorical_crossentropy',
              metrics=['accuracy'])
model.fit(train_images,train_labels,epochs=1,batch_size=128,validation_split=0.1,verbose=1,shuffle=True)
model.evaluate(test_images,test_labels,verbose=1)
# %%
