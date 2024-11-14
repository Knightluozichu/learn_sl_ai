# 超参数
import os


MAX_NUM_TARGETS = 100
# 参数设置
LEARNING_RATE=1e-4
BATCH_SIZE = 1
EPOCHS = 1

BASE_PATH = os.path.dirname(__file__)
ROOT_DIR = os.path.join(BASE_PATH, "data/voc_2007_train_val/VOC2007")
DATA_PATH = os.path.join(BASE_PATH, "data/voc_2007_train_val/VOC2007/JPEGImages")
IMAGES_DIR = os.path.normpath(DATA_PATH)
ANNOTATIONS_DIR = os.path.join(BASE_PATH, "data/voc_2007_train_val/VOC2007/Annotations")
TFRECORD_FILE = os.path.join(BASE_PATH, "data/voc_2007_train_val/VOC2007/train.tfrecord")

# 类别名称和对应的 ID
CLASS_NAMES = ['aeroplane', 'bicycle', 'bird', 'boat', 'bottle',
               'bus', 'car', 'cat', 'chair', 'cow',
               'diningtable', 'dog', 'horse', 'motorbike', 'person',
               'pottedplant', 'sheep', 'sofa', 'train', 'tvmonitor']
CLASS_DICT = {name: idx+1 for idx, name in enumerate(CLASS_NAMES)}
NUM_CLASSES = len(CLASS_NAMES) + 1