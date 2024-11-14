# %%
if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# from requests import get
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn, FasterRCNN_ResNet50_FPN_Weights
from VOC2007_config import CLASS_NAMES, ROOT_DIR
from VOC2007_dataset import get_datasets
import matplotlib.pyplot as plt

# 获取类别数量
num_classes = len(CLASS_NAMES) + 1  # 包括背景类

# 加载预训练的 Faster R-CNN 模型
weights = FasterRCNN_ResNet50_FPN_Weights.COCO_V1
model = fasterrcnn_resnet50_fpn(weights=weights)

# 替换分类头
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = (
    torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
)

# 将模型移动到设备上
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# 获取数据集和数据加载器
train_loader, val_loader = get_datasets()

# 模型path
model_path = os.path.join(os.path.dirname(__file__), "faster_rcnn.pth")

# 检查文件是否存在
if not os.path.exists(model_path):
    raise FileNotFoundError("模型文件不存在，请检查路径。")
else:
    print("模型文件存在")
    # 加载模型
    model.load_state_dict(torch.load(model_path))

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np

# 将模型设置为评估模式
model.eval()

import random
import random

# 获取val_loader的总批次数
total_batches = len(val_loader)

# 随机选择一个批次索引
batch_index = random.randint(0, total_batches - 1)

# 创建一个迭代器
val_iter = iter(val_loader)

# 跳过前面的批次
for _ in range(batch_index):
    next(val_iter)

# 获取随机批次的数据
images, targets = next(val_iter)
# 从验证集获取一个批次的数据
# images, targets = next(iter(val_loader))

# random.seed(42)
# 随机选择一个图像和目标
index = random.randint(0, len(images) - 1)
# 取第一个图像和目标
image = images[index]
target = targets[index]

# 将图像移动到设备上
image = image.to(device)

# 进行预测
with torch.no_grad():
    prediction = model([image])

# 将图像从Tensor转换为numpy数组并调整维度
image = image.cpu().numpy().transpose((1, 2, 0))

# 如果需要，解除标准化（根据实际情况修改）
# image = image * std + mean

# 将图像值从0-1转换为0-255
image = image * 255
image = image.astype(np.uint8)



# 创建绘图
fig, ax = plt.subplots(1, figsize=(12, 8))

# 显示图像
ax.imshow(image)

# 绘制真实的边界框
for box, label in zip(target['boxes'], target['labels']):
    xmin, ymin, xmax, ymax = box.cpu().numpy()
    width, height = xmax - xmin, ymax - ymin
    rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='g', facecolor='none')
    ax.add_patch(rect)
    ax.text(xmin, ymin, CLASS_NAMES[label.item() - 1], fontsize=10, color='g')

# 绘制预测的边界框
for box, score, label in zip(prediction[0]['boxes'], prediction[0]['scores'], prediction[0]['labels']):
    if score > 0.5:
        xmin, ymin, xmax, ymax = box.cpu().numpy()
        width, height = xmax - xmin, ymax - ymin
        rect = patches.Rectangle((xmin, ymin), width, height, linewidth=2, edgecolor='r', facecolor='none')
        ax.add_patch(rect)
        ax.text(xmin, ymin, f"{CLASS_NAMES[label.item() - 1]}: {score:.2f}", fontsize=10, color='r')

plt.show()

# %%
