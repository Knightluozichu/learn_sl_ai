if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import xml.etree.ElementTree as ET
from PIL import Image
import VOC2007_config

class TFRecord_VOC():
    def __init__(self):
        pass

    def create_tf_example(self,img_path, annotation_path):
        # 读取图像
        with tf.io.gfile.GFile(img_path, 'rb') as fid:
            encoded_jpg = fid.read()
        image = Image.open(img_path)
        width, height = image.size

        # 解析 XML 文件
        tree = ET.parse(annotation_path)
        root = tree.getroot()

        filename = root.find('filename').text.encode('utf8')

        xmins = []
        ymins = []
        xmaxs = []
        ymaxs = []
        classes_text = []
        classes = []

        for obj in root.findall('object'):
            # 忽略困难样本
            difficult = int(obj.find('difficult').text)
            if difficult == 1:
                continue

            class_name = obj.find('name').text
            if class_name not in VOC2007_config.CLASS_DICT:
                continue

            classes_text.append(class_name.encode('utf8'))
            classes.append(VOC2007_config.CLASS_DICT[class_name])

            bbox = obj.find('bndbox')
            xmins.append(float(bbox.find('xmin').text) / width)
            ymins.append(float(bbox.find('ymin').text) / height)
            xmaxs.append(float(bbox.find('xmax').text) / width)
            ymaxs.append(float(bbox.find('ymax').text) / height)

        tf_example = tf.train.Example(features=tf.train.Features(feature={
            'image/height': tf.train.Feature(int64_list=tf.train.Int64List(value=[height])),
            'image/width': tf.train.Feature(int64_list=tf.train.Int64List(value=[width])),
            'image/filename': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
            'image/source_id': tf.train.Feature(bytes_list=tf.train.BytesList(value=[filename])),
            'image/encoded': tf.train.Feature(bytes_list=tf.train.BytesList(value=[encoded_jpg])),
            'image/format': tf.train.Feature(bytes_list=tf.train.BytesList(value=['jpeg'.encode('utf8')])),
            'image/object/bbox/xmin': tf.train.Feature(float_list=tf.train.FloatList(value=xmins)),
            'image/object/bbox/xmax': tf.train.Feature(float_list=tf.train.FloatList(value=xmaxs)),
            'image/object/bbox/ymin': tf.train.Feature(float_list=tf.train.FloatList(value=ymins)),
            'image/object/bbox/ymax': tf.train.Feature(float_list=tf.train.FloatList(value=ymaxs)),
            'image/object/class/text': tf.train.Feature(bytes_list=tf.train.BytesList(value=classes_text)),
            'image/object/class/label': tf.train.Feature(int64_list=tf.train.Int64List(value=classes)),
        }))
        return tf_example

    # 将数据集转换为 TFRecord
    def convert_to_tfrecord(self):
        writer = tf.io.TFRecordWriter(VOC2007_config.TFRECORD_FILE)
        image_files = os.listdir(VOC2007_config.IMAGES_DIR)
        for idx, img_name in enumerate(image_files):
            img_path = os.path.join(VOC2007_config.IMAGES_DIR, img_name)
            annotation_path = os.path.join(VOC2007_config.ANNOTATIONS_DIR, img_name.replace('.jpg', '.xml'))
            if not os.path.exists(annotation_path):
                continue
            tf_example = self.create_tf_example(img_path, annotation_path)
            writer.write(tf_example.SerializeToString())
            if idx % 100 == 0:
                print(f'已处理 {idx} 张图像')
        writer.close()
        # print('TFRecord 文件生成完成')

    # 读取 TFRecord 文件
    def parse_tfrecord(self,example_proto):
        feature_description = {
            'image/height': tf.io.FixedLenFeature([], tf.int64),
            'image/width': tf.io.FixedLenFeature([], tf.int64),
            'image/filename': tf.io.FixedLenFeature([], tf.string),
            'image/source_id': tf.io.FixedLenFeature([], tf.string),
            'image/encoded': tf.io.FixedLenFeature([], tf.string),
            'image/format': tf.io.FixedLenFeature([], tf.string),
            'image/object/bbox/xmin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/xmax': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymin': tf.io.VarLenFeature(tf.float32),
            'image/object/bbox/ymax': tf.io.VarLenFeature(tf.float32),
            'image/object/class/text': tf.io.VarLenFeature(tf.string),
            'image/object/class/label': tf.io.VarLenFeature(tf.int64),
        }
        parsed_example = tf.io.parse_single_example(example_proto, feature_description)

        image = tf.image.decode_jpeg(parsed_example['image/encoded'], channels=3)

        # 获取边界框和标签
        xmin = tf.sparse.to_dense(parsed_example['image/object/bbox/xmin'])
        xmax = tf.sparse.to_dense(parsed_example['image/object/bbox/xmax'])
        ymin = tf.sparse.to_dense(parsed_example['image/object/bbox/ymin'])
        ymax = tf.sparse.to_dense(parsed_example['image/object/bbox/ymax'])
        labels = tf.sparse.to_dense(parsed_example['image/object/class/label'])
        boxes = tf.stack([ymin, xmin, ymax, xmax], axis=1)

        return image, boxes, labels


# 测试
if __name__ == '__main__':
    tfrecord_voc = TFRecord_VOC()
    # 存储为 TFRecord 文件
    tfrecord_voc.convert_to_tfrecord()
    print('TFRecord 文件生成完成')

    # 读取 TFRecord 文件
    tfrecord_file = tf.data.TFRecordDataset(VOC2007_config.TFRECORD_FILE)
    dataset = tfrecord_file.shuffle(buffer_size=1000)
    dataset = dataset.map(tfrecord_voc.parse_tfrecord, num_parallel_calls=tf.data.AUTOTUNE)
    print(dataset)
    print('TFRecord 文件读取完成')

# %%
# import numpy as np

# image = np.random.randint(0, 256, size=(224, 224, 3))
# print(image.shape)
# print(image.shape[:2])
# print(image.shape[-2:])
# %%
