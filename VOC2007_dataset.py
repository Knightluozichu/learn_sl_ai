# %% torchdata
import os
import torch
from torch.utils.data import Dataset
import xml.etree.ElementTree as ET
from PIL import Image

import torch
import torchvision.transforms.functional as F
import random

from learn_sl_ai.VOC2007_config import ROOT_DIR

class ComposeDouble(object):
    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target

class ToTensorDouble(object):
    def __call__(self, image, target):
        image = F.to_tensor(image)
        return image, target

class RandomHorizontalFlipDouble(object):
    def __init__(self, p=0.5):
        self.p = p

    def __call__(self, image, target):
        if random.random() < self.p:
            image = F.hflip(image)
            width = image.shape[-1]
            bbox = target['boxes']
            bbox[:, [0, 2]] = width - bbox[:, [2, 0]]  # 水平翻转边界框
            target['boxes'] = bbox
        return image, target
class VOCDataset(Dataset):
    def __init__(self, root_dir, image_set="train", transforms=None):
        self.root_dir = root_dir
        self.image_set = image_set
        self.transforms = transforms
        print("Type of transforms:", type(self.transforms))
        # 图像和标注的路径
        self.image_dir = os.path.join(root_dir, "JPEGImages")
        self.annotation_dir = os.path.join(root_dir, "Annotations")

        # 图像索引列表
        image_sets_file = os.path.join(
            root_dir, "ImageSets", "Main", f"{image_set}.txt"
        )
        with open(image_sets_file) as f:
            self.image_ids = [line.strip() for line in f]

        # 类别名称和索引的映射
        self.class_names = ["__background__"] + [
            "aeroplane",
            "bicycle",
            "bird",
            "boat",
            "bottle",
            "bus",
            "car",
            "cat",
            "chair",
            "cow",
            "diningtable",
            "dog",
            "horse",
            "motorbike",
            "person",
            "pottedplant",
            "sheep",
            "sofa",
            "train",
            "tvmonitor",
        ]
        self.class_dict = {name: idx for idx, name in enumerate(self.class_names)}

    def __len__(self):
        return len(self.image_ids)

    def __getitem__(self, idx):
        # 获取图像 ID
        image_id = self.image_ids[idx]

        # 加载图像
        image_path = os.path.join(self.image_dir, f"{image_id}.jpg")
        image = Image.open(image_path).convert("RGB")

        # 加载标注
        annotation_path = os.path.join(self.annotation_dir, f"{image_id}.xml")
        boxes = []
        labels = []
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        for obj in root.findall("object"):
            # 忽略困难样本
            if int(obj.find("difficult").text) == 1:
                continue

            # 获取边界框
            bbox = obj.find("bndbox")
            xmin = float(bbox.find("xmin").text) - 1
            ymin = float(bbox.find("ymin").text) - 1
            xmax = float(bbox.find("xmax").text) - 1
            ymax = float(bbox.find("ymax").text) - 1
            boxes.append([xmin, ymin, xmax, ymax])

            # 获取类别标签
            label = self.class_dict[obj.find("name").text.lower().strip()]
            labels.append(label)

        # 将数据转换为张量
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        # 创建 target 字典
        target = {}
        target["boxes"] = boxes
        areas = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        target['area'] = torch.as_tensor(areas, dtype=torch.float32)
        # 添加 'iscrowd' 字段，默认为 0（表示没有遮挡）
        target['iscrowd'] = torch.zeros((boxes.shape[0],), dtype=torch.int64)
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])

        # 如果需要，可以添加 masks、area、iscrowd 等字段
        # print(self.transforms)
        # print(image, target)

        # 应用数据增强（如果有）
        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return image, target

# %%
# import torchvision.transforms as T

# 修改 get_transform 函数
def get_transform(train):
    transforms = []
    # 使用我们自定义的 ToTensorDouble
    transforms.append(ToTensorDouble())
    if train:
        # 使用我们自定义的 RandomHorizontalFlipDouble
        transforms.append(RandomHorizontalFlipDouble(0.5))
    # 使用我们自定义的 ComposeDouble
    return ComposeDouble(transforms)


from sklearn.model_selection import train_test_split


def get_datasets():
    # 创建数据集
    train_dataset = VOCDataset(ROOT_DIR, transforms=get_transform(train=True))
    val_dataset = VOCDataset(ROOT_DIR, transforms=get_transform(train=False))
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # 创建数据加载器 - 设置 num_workers=0 以获取详细错误信息
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=0,  # 修改为0
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=0,  # 修改为0
        collate_fn=collate_fn
    )

    return train_loader, val_loader

#  %%
# 测试代码修改
import torch
from torch.utils.data import DataLoader
import traceback

try:
    # 创建数据集
    train_dataset = VOCDataset(ROOT_DIR, transforms=get_transform(train=True))
    val_dataset = VOCDataset(ROOT_DIR, transforms=get_transform(train=False))
    
    def collate_fn(batch):
        return tuple(zip(*batch))
    
    # 创建数据加载器 - 设置 num_workers=0 以获取详细错误信息
    train_loader = DataLoader(
        train_dataset, 
        batch_size=2, 
        shuffle=True, 
        num_workers=0,  # 修改为0
        collate_fn=collate_fn
    )
    
    val_loader = DataLoader(
        val_dataset, 
        batch_size=2, 
        shuffle=False, 
        num_workers=0,  # 修改为0
        collate_fn=collate_fn
    )

    # 添加调试信息
    print("Dataset size:", len(train_dataset))
    
    # 测试单个数据项
    print("\nTesting single item fetch:")
    sample_image, sample_target = train_dataset[0]
    print("Sample image shape:", sample_image.shape if torch.is_tensor(sample_image) else "Not a tensor")
    print("Sample target:", sample_target)
    
    # 测试数据加载器
    print("\nTesting dataloader:")
    for i, (images, targets) in enumerate(train_loader):
        print(f"\nBatch {i}:")
        print("Images:", [img.shape for img in images])
        print("Targets:", targets)
        if i >= 0:  # 只打印第一个批次
            break

except Exception as e:
    print("Error occurred:")
    print(traceback.format_exc())
# %%

# # %% tfdata
# import tensorflow as tf
# import VOC2007_config
# from TFRecord_voc2007 import TFRecord_VOC


# class Dataset_VOC:
#     def __init__(self, tfrecord):
#         self.tfrecord = tfrecord

#     # 数据预处理函数
#     def preprocess_data(self, image, boxes, labels):
#         # 调整图像大小
#         image = tf.image.resize(image, (224, 224))
#         image = tf.cast(image, tf.float32) / 255.0

#         # 如果需要边界框以像素为单位，可以取消注释以下代码
#         # boxes = boxes * 224

#         # 如果保持边界框归一化，则无需修改 boxes

#         return image, boxes, labels

#     # 填充目标函数
#     def pad_targets(self, image, boxes, labels):
#         # 确保 boxes 和 labels 数量一致
#         num_boxes = tf.shape(boxes)[0]
#         num_labels = tf.shape(labels)[0]
#         tf.debugging.assert_equal(
#             num_boxes, num_labels, message="Boxes 和 Labels 的数量不匹配"
#         )

#         # 如果目标数量超过 MAX_NUM_TARGETS，则截断
#         boxes = boxes[: VOC2007_config.MAX_NUM_TARGETS, :]
#         labels = labels[: VOC2007_config.MAX_NUM_TARGETS]

#         # 填充 boxes
#         padded_boxes = tf.pad(
#             boxes,
#             [[0, VOC2007_config.MAX_NUM_TARGETS - tf.shape(boxes)[0]], [0, 0]],
#             constant_values=0.0,
#         )

#         # 填充 labels
#         padded_labels = tf.pad(
#             labels,
#             [[0, VOC2007_config.MAX_NUM_TARGETS - tf.shape(labels)[0]]],
#             constant_values=0,
#         )

#         return image, padded_boxes, padded_labels

#     # 生成 Dataset 数据集
#     def create_dataset(self, tfrecord_file, shuffle_buffer=1000):
#         dataset = tf.data.TFRecordDataset(tfrecord_file)
#         dataset = dataset.shuffle(buffer_size=shuffle_buffer)
#         dataset = dataset.map(
#             self.tfrecord.parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE
#         )
#         # 数据预处理
#         dataset = dataset.map(self.preprocess_data, num_parallel_calls=tf.data.AUTOTUNE)
#         # 填充目标
#         dataset = dataset.map(self.pad_targets, num_parallel_calls=tf.data.AUTOTUNE)
#         # 移除批处理操作
#         # dataset = dataset.batch(batch_size)
#         dataset = dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#         return dataset

#     def split_dataset(self):
#         # 创建 Dataset 数据集
#         dataset = self.create_dataset(VOC2007_config.TFRECORD_FILE)

#         # 计算数据集大小
#         total_size = dataset.reduce(0, lambda x, _: x + 1).numpy()
#         print(f"数据集总大小：{total_size}")

#         # 划分训练集和验证集
#         train_size = int(total_size * 0.8)
#         val_size = total_size - train_size

#         train_dataset = dataset.take(train_size)
#         val_dataset = dataset.skip(train_size)

#         # 批处理并填充
#         train_dataset = train_dataset.padded_batch(
#             VOC2007_config.BATCH_SIZE,
#             padded_shapes=(
#                 [224, 224, 3],  # 图像形状
#                 [VOC2007_config.MAX_NUM_TARGETS, 4],  # 边界框形状
#                 [VOC2007_config.MAX_NUM_TARGETS],  # 标签形状
#             ),
#             padding_values=(
#                 0.0,  # 图像填充值
#                 0.0,  # 边界框填充值
#                 tf.constant(0, dtype=tf.int64),  # 标签填充值
#             ),
#             drop_remainder=True,
#         )

#         val_dataset = val_dataset.padded_batch(
#             VOC2007_config.BATCH_SIZE,
#             padded_shapes=(
#                 [224, 224, 3],
#                 [VOC2007_config.MAX_NUM_TARGETS, 4],
#                 [VOC2007_config.MAX_NUM_TARGETS],
#             ),
#             padding_values=(0.0, 0.0, tf.constant(0, dtype=tf.int64)),
#             drop_remainder=True,
#         )

#         # 预取
#         train_dataset = train_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)
#         val_dataset = val_dataset.prefetch(buffer_size=tf.data.AUTOTUNE)

#         # 验证数据集
#         print("训练集示例：")
#         for images, boxes, labels in train_dataset.take(1):
#             print(f"图像形状: {images.shape}")
#             print(f"边界框形状: {boxes.shape}")
#             print(f"标签形状: {labels.shape}")

#         print("验证集示例：")
#         for images, boxes, labels in val_dataset.take(1):
#             print(f"图像形状: {images.shape}")
#             print(f"边界框形状: {boxes.shape}")
#             print(f"标签形状: {labels.shape}")

#         return train_dataset, val_dataset


# def get_dataset():
#     dataset_voc = Dataset_VOC(TFRecord_VOC())
#     train_dataset, val_dataset = dataset_voc.split_dataset()
#     return train_dataset, val_dataset


# # 测试

# dataset_voc = Dataset_VOC(TFRecord_VOC())
# train_dataset, val_dataset = dataset_voc.split_dataset()
# for images, boxes, labels in train_dataset.take(1):
#     print(f"Training batch - images shape: {images.shape}")
#     print(f"Training batch - boxes shape: {boxes.shape}")
#     print(f"Training batch - labels shape: {labels.shape}")
# print("数据集划分完成")
# # %%
# import matplotlib.pyplot as plt

# for images, boxes, labels in train_dataset.take(1):
#     image = images[0].numpy()
#     bbox = boxes[0].numpy()
#     label = labels[0].numpy()
#     plt.imshow(image)
#     for i in range(len(bbox)):
#         ymin, xmin, ymax, xmax = bbox[i]
#         if label[i] == 0:
#             # 填充的边界框使用蓝色
#             edgecolor = "blue"
#         else:
#             # 有效的边界框使用红色
#             edgecolor = "red"
#         rect = plt.Rectangle(
#             (xmin * 224, ymin * 224),
#             (xmax - xmin) * 224,
#             (ymax - ymin) * 224,
#             fill=False,
#             color=edgecolor,
#             linewidth=2,
#         )
#         plt.gca().add_patch(rect)
#     plt.show()
