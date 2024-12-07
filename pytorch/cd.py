# %%
from turtle import title
import requests
import zipfile
from pathlib import Path

import torch.utils.data.dataloader

from learn_sl_ai.VOC2007_config import EPOCHS

# from pytorch.data_loader import ImageFolderCustom

data_path = Path("data/")
image_path = data_path / "pizza_steak_sushi"

if image_path.is_dir():
    print(f"{image_path} directory exists.")
else:
    print(f"Did not find {image_path} directory, creating one...")
    image_path.mkdir(parents=True,exist_ok=True)

    # with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
    #     request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
    #     f.write(request.content)
    from tqdm import tqdm

    def download_with_progress(url, save_path):
        response = requests.get(url, stream=True)
        total_size = int(response.headers.get('content-length', 0))
        with open(save_path, "wb") as f, tqdm(
            desc=save_path.name,
            total=total_size,
            unit='iB',
            unit_scale=True,
        ) as bar:
            for data in response.iter_content(chunk_size=1024):
                size = f.write(data)
                bar.update(size)

    download_with_progress("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip", data_path / "pizza_steak_sushi.zip")
        
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip","r") as zip_ref:
        print("Unzipping pizza, steak, suzhi data ...")
        zip_ref.extractall(image_path)
    
# %%
import os
def walk_through_dir(dir_path):
    for dirpath,dirnames, filenames in os.walk(dir_path):
        print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}")

walk_through_dir(image_path)
# %%
# 设置训练和测试路径
train_dir = image_path / "train"
test_dir = image_path / "test"

train_dir, test_dir
# %%
# 可视化图像
import random
from PIL import Image

random.seed(42)

image_path_list = list(image_path.glob("*/*/*.jpg"))
random_image_path = random.choice(image_path_list)
image_class = random_image_path.parent.stem
img = Image.open(random_image_path)

print(f"Random image path:{random_image_path}")
print(f"Image class:{image_class}")
print(f"Image height:{img.height}")
print(f"Image width:{img.width}")
img
# %%
import numpy as np
import matplotlib.pyplot as plt

img_as_array = np.asarray(img)

plt.figure(figsize=(10,7))
plt.imshow(img_as_array)
plt.title(f"Image class:{image_class} | Image shape:{img_as_array}")
plt.axis(False)

# %%
import torch
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

data_transform = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

def plot_transformed_images(image_paths,transform,n=3,seed=42):
    random.seed(seed)
    random_image_paths = random.sample(image_paths,k=n)
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
            fig, ax = plt.subplots(1, 2)
            ax[0].imshow(f)
            ax[0].set_title(f"Original \nSzie:{f.size}")
            ax[0].axis("off")
            
            transformed_image = transform(f).permute(1,2,0)
            ax[1].imshow(transformed_image)
            ax[1].set_title(f"Transformed \nSize:{transformed_image.shape}")
            ax[1].axis("off")
            
            fig.suptitle(f"Class : {image_path.parent.stem}")
    
plot_transformed_images(image_path_list,
                        transform=data_transform,
                        n=3)

# %%
# 使用 ImageFolder 加载图像数据
from torchvision import datasets
train_data = datasets.ImageFolder(root=train_dir,
                                  transform=data_transform,
                                  target_transform=None)
test_data = datasets.ImageFolder(root=test_dir,
                                 transform=data_transform,
                                 target_transform=None)
# print(f"Train data:\n{train_data} \nTest data :\n{test_data}")
# %%
class_names = train_data.classes
class_names
# %%
class_dict = train_data.class_to_idx
class_dict
# %%
len(train_data), len(test_data)
# %%
img, label = train_data[0][0], train_data[0][1]
print(f"Image tensor:\n{img}")
print(f"Image shape:{img.shape}")
print(f"Image datatype:{img.dtype}")
print(f"Image Label:{label}")
print(f"Label datatype:{type(label)}")
# %%
img_permute = img.permute(1,2,0)

print(f"Original shape:{img.shape} -> [color_channels, height, width]")
print(f"Image permute shape:{img_permute.shape} -> [height,width,color_channels]")

plt.figure(figsize=(10,7))
plt.imshow(img_permute)
plt.axis("off")
plt.title(class_names[label], fontsize=14)

# %%
# 将加载的图片转换为 DataLoader
# 我们将使用 torch.utils.data.DataLoader 来做到这一点
from torch.utils.data import DataLoader
train_dataloader = DataLoader(dataset=train_data,
                              batch_size=1,
                              num_workers=1,
                              shuffle=True)

test_dataloader = DataLoader(dataset=test_data,
                             batch_size=1,
                             num_workers=1,
                             shuffle=False)

try:
    img, label = next(iter(train_dataloader))
    print(f"Image shape: {img.shape} -> [batch_size, color_channels, height, width]")
    print(f"Label shape: {label.shape}")
except Exception as e:
    print(f"Error occurred while loading data: {e}")
# %%
import os
import pathlib
import torch

from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
from typing import Tuple,Dict, List

train_data.classes, train_data.class_to_idx
# %%
# 创建辅助函数来获取类名
target_directory = train_dir
print(f"Target directory :{target_directory}")

class_names_found = sorted([entry.name for entry in list(os.scandir(image_path/"train"))])
print(f"Class names found:{class_names_found}")

# %%
def find_classes(directory:str) -> Tuple[List[str], Dict[str,int]]:
    classes = sorted(entry.name for entry in os.scandir(directory) if entry.is_dir())
    if not classes:
        raise FileNotFoundError(f"Couldn't find any classes in {directory}.")
    class_to_idx = {cls_name:i for i, cls_name in enumerate(classes)}
    return classes, class_to_idx

find_classes(train_dir)
# %%

if '__file__' in globals():
    import os, sys
    sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from pytorch.data_loader import ImageFolderCustom

# 创建自定义 Dataset 以复制 ImageFolder
train_transforms = transforms.Compose([
    transforms.Resize((64, 64)),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

train_data_custom = ImageFolderCustom(targ_dir=train_dir,
                                      transform=train_transforms)
test_data_custom = ImageFolderCustom(targ_dir=test_dir,
                                     transform=test_transforms)
# %%
len(train_data_custom), len(test_data_custom)
# %%
train_data_custom.classes, train_data_custom.class_to_idx
# %%
print((len(train_data_custom) == len(train_data)) & (len(test_data_custom) == len(test_data)))
print(train_data_custom.classes == train_data.classes)
print(train_data_custom.class_to_idx == train_data.class_to_idx)

# %%
def display_random_images(dataset:torch.utils.data.dataset.Dataset,
                          classes:List[str] = None,
                          n:int = 10,
                          display_shape:bool = True,
                          seed:int = None):
    if n > 10:
        n = 10
        display_shape = False
        print(f"For display purposes, n shouldn't be larger than 10, setting to 10 and removing shape display.")
    
    if seed:
        random.seed(seed)
        
    random_samples_idx = random.sample(range(len(dataset)), k=n)
    plt.figure(figsize=(16,8))
    
    for i,targ_sample in enumerate(random_samples_idx):
        targ_image, targ_label = dataset[targ_sample][0],dataset[targ_sample][1]
        
        targ_image_adjust = targ_image.permute(1, 2, 0)
        
        plt.subplot(1, n, i+1)
        plt.imshow(targ_image_adjust)
        plt.axis("off")
        if classes:
            title = f"class{classes[targ_label]}"
            if display_shape:
                title = title + f"\nshape:{targ_image_adjust.shape}"
        plt.title(title)

display_random_images(train_data,
                      n=5,
                      classes=class_names,
                      seed=None)
# %%
display_random_images(train_data,
                      n=12,
                      classes=class_names,
                      seed=None)
# %%
# 如何将自定义Dataset转换为 DataLoader 数据集？
# 使用 torch.utils.data.DataLoader()
# Turn train and test custom Dataset's into DataLoader's
from torch.utils.data import DataLoader
train_dataloader_custom = DataLoader(dataset=train_data_custom, # use custom created train Dataset
                                     batch_size=1, # how many samples per batch?
                                     num_workers=0, # how many subprocesses to use for data loading? (higher = more)
                                     shuffle=True) # shuffle the data?

test_dataloader_custom = DataLoader(dataset=test_data_custom, # use custom created test Dataset
                                    batch_size=1, 
                                    num_workers=0, 
                                    shuffle=False) # don't usually need to shuffle testing data

train_dataloader_custom, test_dataloader_custom
# %%
# Get image and label from custom DataLoader
img_custom, label_custom = next(iter(train_dataloader_custom))

# Batch size will now be 1, try changing the batch_size parameter above and see what happens
print(f"Image shape: {img_custom.shape} -> [batch_size, color_channels, height, width]")
print(f"Label shape: {label_custom.shape}")
# %%
from torchvision import transforms

train_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transforms = transforms.Compose([
    transforms.Resize((224,224)),
    transforms.ToTensor()
])

image_path_list = list(image_path.glob("*/*/*.jpg"))
plot_transformed_images(
    image_paths=image_path_list,
    transform=train_transforms,
    n=3,
    seed=None,
)
# %%
# 1.为Model0创建transform并加载数据
simple_transform = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

from torchvision import datasets
train_data_simple = datasets.ImageFolder(root=train_dir, transform=simple_transform)
test_data_simple = datasets.ImageFolder(root=test_dir,transform=simple_transform)

import os
from torch.utils.data import DataLoader

BATCH_SIZE=32
# NUM_WORKERS=os.cpu_count()
NUM_WORKERS = 0
print(f"Creating DataLoader's with batch size {BATCH_SIZE}")

train_dataloader_simple = DataLoader(train_data_simple,
                                     batch_size=BATCH_SIZE,
                                     shuffle=True,
                                     num_workers=NUM_WORKERS)
test_dataloader_simple = DataLoader(dataset=test_data_simple,
                                    batch_size=BATCH_SIZE,
                                    num_workers=NUM_WORKERS,
                                    shuffle=False)
train_dataloader_simple, test_dataloader_simple

# %%
# 2.创建TinyVGG模型类
import torch.nn as nn
class MyTinyVGG(nn.Module):
    def __init__(self, input_shape, hidden_units,output_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape,hidden_units,(3,3),
                      stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units,out_channels=hidden_units,
                      kernel_size=(3,3), stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=2,padding=0,dilation=1,ceil_mode=False)
        )
        
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units,
                      3,1,1),
            nn.ReLU(),
            nn.Conv2d(hidden_units,hidden_units,
                      3,1,1),
            nn.ReLU(),
            nn.MaxPool2d(2,2,0,1,False)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_units * 16 * 16, out_features=
                      output_shape,bias=True)
        )
        
    def forward(self, X:torch.Tensor):
        X = self.block_1(X)
        X = self.block_2(X)
        X = self.classifier(X)
        return X

torch.manual_seed(42)
model_0 = MyTinyVGG(3,10,3)
model_0
# %%
# img_batch, label_batch = None,None
if __name__ == "__main__":
    device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
    try:
        model_0.to(device)
        # train_dataloader
        img_batch, label_batch = next(iter(train_dataloader_simple))
        img_single, label_single = img_batch[0].unsqueeze(dim=0), label_batch[0]
        img_single, label_single = img_single.to(device), label_single.to(device)
        print(f"Single image shape:{img_single.shape}")
        model_0.eval()
        with torch.inference_mode():
            pred = model_0(img_single)
        print(f"Output logits:\n{pred}\n")
        print(f"Output prediction probabilities:\n{torch.softmax(pred, dim=1)}\n")
        print(f"Output prediction label:\n{torch.argmax(torch.softmax(pred, dim=1))}\n")
        print(f"Actual label:\n{label_single}")
    except Exception as e:
        print(f"e:{e}")
# %%
# Install torchinfo if it's not available, import it if it is
import torchinfo
# try: 
#     import torchinfo
# except:
#     !pip install torchinfo
#     import torchinfo
    
from torchinfo import summary
summary(model_0, input_size=[1, 3, 64, 64]) # do a test pass through of an example input size 
# %%
def train_step(model:torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer):
    model.train()
    train_loss, train_acc = 0,0
    
    for batch,(X,y) in enumerate(dataloader):
        X,y = X.to(device),y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        y_pred_class = torch.argmax(torch.softmax(y_pred,dim=1),dim=1)
        train_acc += (y_pred_class == y).sum().item() / len(y_pred)
        
    train_loss = train_loss/len(dataloader)
    train_acc = train_acc / len(dataloader)
    
    return train_loss, train_acc

def test_step(model:torch.nn.Module,
              dataloader:torch.utils.data.DataLoader,
              loss_fn:torch.nn.Module,
              device:torch.device
              ):
    model.to(device)
    model.eval()
    test_loss,  test_acc = 0,0 
    with torch.inference_mode():
        for X,y in dataloader:
            X,y = X.to(device), y.to(device)
            test_pred = model(X)
            test_loss += loss_fn(test_pred,y).item()
            y_pred_class = torch.argmax(torch.softmax(test_pred,dim=1),dim=1)
            test_acc += (y_pred_class == y).sum().item() / len(test_pred)
    test_loss /= len(dataloader)
    test_acc /= len(dataloader)
    return test_loss, test_acc
            

from tqdm.auto import tqdm

def train(model:torch.nn.Module,
          train_dataloader:torch.utils.data.DataLoader,
          test_dataloader:torch.utils.data.DataLoader,
          optimizer:torch.optim.Optimizer,
          loss_fn:torch.nn.Module = nn.CrossEntropyLoss(),
          epochs:int=5):
    results = {"train_loss":[],
               "train_acc":[],
               "test_loss":[],
               "test_acc":[]}
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(
            model=model,
            dataloader=train_dataloader,
            loss_fn=loss_fn,
            optimizer=optimizer
        )
        test_loss,test_acc  =test_step(
            model=model,
            dataloader=test_dataloader,
            loss_fn=loss_fn
        )
        print(
            f"Epoch:{epoch+1} | "
            f"train_loss:{train_loss:.4f} | "
            f"train_acc:{train_acc:.4f} | "
            f"test_loss:{test_loss:.4f} | "
            f"test_acc:{test_acc:.4f}"
        )
        
        results["train_loss"].append(train_loss.item() if isinstance(train_loss, torch.Tensor) else train_loss)
        results["train_acc"].append(train_acc.item() if isinstance(train_acc, torch.Tensor) else train_acc)
        results["test_loss"].append(test_loss.item() if isinstance(test_loss, torch.Tensor) else test_loss)
        results["test_acc"].append(test_acc.item() if isinstance(test_acc, torch.Tensor) else test_acc)
        
    return results
# %%
import multiprocessing

if __name__ == "__main__":
    multiprocessing.freeze_support()
    torch.manual_seed(42)
    torch.mps.manual_seed(42)
    NUM_EPOCHS = 20
    model_0 = MyTinyVGG(
        input_shape=3,
        hidden_units=10,
        output_shape=len(train_data.classes)
    ).to(device)
    loss_fn = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(params=model_0.parameters(),lr=1e-3)
    from timeit import default_timer as timer
    start_time = timer()
    try:
        model_0_results = train(model=model_0,
                            train_dataloader=train_dataloader_simple,
                            test_dataloader=test_dataloader_simple,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=NUM_EPOCHS)
    except Exception as e:
        print(f"e:{e}")
    end_time = timer()
    print(f"Total training time:{end_time-start_time:.3f} seconds")
# %%
def plot_loss_curves(results:Dict[str,List[float]]):
    loss = results['train_loss']
    test_loss = results['test_loss']
    
    accuracy = results['train_acc']
    test_accuracy = results['test_acc']
    
    epochs = range(len(results['train_loss']))
    
    plt.subplot(1,2,1)
    plt.plot(epochs,loss,label='train_loss')
    plt.plot(epochs,test_loss,label='test_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.legend()
    
    plt.subplot(1,2,2)
    plt.plot(epochs, accuracy, label='train_accuracy')
    plt.plot(epochs,test_accuracy,label='test_accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.legend()


plot_loss_curves(model_0_results)
    
# %%
train_transform_trivial_augment = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])

test_transform_trivial_augment = transforms.Compose([
    transforms.Resize((64,64)),
    transforms.ToTensor()
])

train_data_augmented = datasets.ImageFolder(root=train_dir,
                                            transform=train_transform_trivial_augment
                                            )
test_data_simple = datasets.ImageFolder(root=test_dir,
                                           transform=test_transform_trivial_augment)

import os
BATCH_SIZE=32
NUM_WORKERS = 0
torch.manual_seed(42)
train_dataloader_augmented = DataLoader(train_data_augmented,
                                        batch_size=BATCH_SIZE,
                                        shuffle=True,
                                        num_workers=NUM_WORKERS)
test_dataloader_simple = DataLoader(test_data_simple,
                                    BATCH_SIZE,
                                    False,
                                    num_workers=NUM_WORKERS)

train_dataloader_augmented, test_dataloader
# %%
torch.manual_seed(42)
model_1 = MyTinyVGG(input_shape=3,
                    hidden_units=10,
                    output_shape=len(train_data_augmented.classes)).to(device)
model_1
# %%
torch.manual_seed(42)
torch.mps.manual_seed(42)

NUM_EPOCHS = 15

loss_fn = nn.CrossEntropyLoss()

optimizer = torch.optim.Adam(params=model_1.parameters(),lr=0.001)

from timeit import default_timer as timer
start_time = timer()

model_1_results = train(model=model_1,
                        train_dataloader=train_dataloader_augmented,
                        test_dataloader=test_dataloader_simple,
                        optimizer=optimizer,
                        loss_fn=loss_fn,
                        epochs=NUM_EPOCHS)
end_time = timer()
print(f"Total training time:{end_time-start_time:.3f} seconds")
# %%
plot_loss_curves(model_1_results)
# %%
# 比较模型结果
import pandas as pd
model_0_df = pd.DataFrame(model_0_results)
model_1_df = pd.DataFrame(model_1_results)
model_0_df
# %%
import requests
custom_image_path = data_path / "04-pizza-dad.jpeg"

if not custom_image_path.is_file():
    with open(custom_image_path, "wb") as f:
        request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")

# %%
import torchvision

custom_image_uint8 = torchvision.io.read_image(str(custom_image_path))
# Print out image data
print(f"Custom image tensor:\n{custom_image_uint8}\n")
print(f"Custom image shape: {custom_image_uint8.shape}\n")
print(f"Custom image dtype: {custom_image_uint8.dtype}")
# %%
# Try to make a prediction on image in uint8 format (this will error)
model_1.eval()
with torch.inference_mode():
    model_1(custom_image_uint8.to(device))
# %%
custom_image = torchvision.io.read_image(str(custom_image_path)).to(dtype=torch.float32)
custom_image /= 255.
# Print out image data
print(f"Custom image tensor:\n{custom_image}\n")
print(f"Custom image shape: {custom_image.shape}\n")
print(f"Custom image dtype: {custom_image.dtype}")
# %%
plt.imshow(custom_image.permute(1,2,0))
plt.title(f"Image shape:{custom_image.shape}")
plt.axis(False)
# %%
custom_image_transform = transforms.Compose([
    transforms.Resize((64,64)),
])
custom_image_transformed = custom_image_transform(custom_image)
print(f"Original shape: {custom_image.shape}")
print(f"New shape: {custom_image_transformed.shape}")
# %%
model_1.eval()
with torch.inference_mode():
    custom_image_transformed_with_bs = custom_image_transformed.unsqueeze(dim=0)
    custom_image_pred = model_1(custom_image_transformed.unsqueeze(dim=0).to(device))
# %%
# 注意：我们刚刚经历了三个经典且最常见的深度学习和 PyTorch 问题：

# Wrong datatypes - our model expects torch.float32 where our original custom image was uint8.
# 数据类型错误 - 我们的模型需要 torch.float32，而我们最初的自定义图像是 uint8。
# Wrong device - our model was on the target device (in our case, the GPU) whereas our target data hadn't been moved to the target device yet.
# 错误的设备 - 我们的模型位于目标device（在本例中为 GPU）上，而我们的目标数据尚未移动到目标device。
# Wrong shapes - our model expected an input image of shape [N, C, H, W] or [batch_size, color_channels, height, width] whereas our custom image tensor was of shape [color_channels, height, width].
# 错误的形状 - 我们的模型期望输入图像为形状 [N, C, H, W] 或 [batch_size, color_channels, height, width]而我们的自
# Print out prediction logits
print(f"Prediction logits: {custom_image_pred}")

# Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
custom_image_pred_probs = torch.softmax(custom_image_pred, dim=1)
print(f"Prediction probabilities: {custom_image_pred_probs}")

# Convert prediction probabilities -> prediction labels
custom_image_pred_label = torch.argmax(custom_image_pred_probs, dim=1)
print(f"Prediction label: {custom_image_pred_label}")
# %%
custom_image_pred_class = class_names[custom_image_pred_label]
custom_image_pred_class
# %%
def pred_and_plot_image(model:torch.nn.Module,
                        image_path:str,
                        class_names:List[str] = None,
                        transform=None,
                        device:torch.device=device):
    target_image = torchvision.io.read_image(str(image_path)).type(torch.float32)
    target_image  =  target_image / 255.
    if transform:
        target_image  = transform(target_image)
    
    model.to(device)
    model.eval()
    with torch.inference_mode():
        target_image = target_image.unsqueeze(dim=0)
        
        target_image_pred = model(target_image.to(device))
        
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)
    target_image_pred_label = torch.argmax(target_image_pred_probs,dim=1)
    
    plt.imshow(target_image.squeeze().permute(1,2,0))
    if class_names:
        title = f"Pred: {class_names[target_image_pred_label.cpu()]} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    else: 
        title = f"Pred: {target_image_pred_label} | Prob: {target_image_pred_probs.max().cpu():.3f}"
    plt.title(title)
    plt.axis(False)

pred_and_plot_image(model=model_1,
                    image_path=custom_image_path,
                    class_names=class_names,
                    transform=custom_image_transform,
                    device=device)

# %%
# 我们的模型表现不佳（数据拟合不佳）。防止欠拟合的 3 种方法是什么？将它们写下来并用一句话解释每个。
# 1.增加样本数据，数据量不够，模型没有学习到重要特征
# 2.向模型添加更多层/单元 它可能没有足够的能力来学习数据所需的模式/权重/表示以进行预测。为模型添加更多预测能力的一种方法是增加这些层中隐藏层/单元的数量。
# 3.调整学习率 也许您的模型的学习率太高，无法开始。它试图在每个 epoch 中过多地更新其权重，反过来却没有学习任何东西。在这种情况下，您可能会降低学习率并查看会发生什么。
# 4.增加EPOCHS 有时，模型只是需要更多时间来学习数据的表示形式。如果你发现在较小的实验中，你的模型没有学习任何东西，也许让它训练更多的 epoch 可能会带来更好的性能。
# 5.使用迁移学习 迁移学习能够防止过拟合和欠拟合。它涉及使用以前工作模型中的模式，并根据您自己的问题调整它们
# 6.使用更少的正则化 也许您的模型欠拟合是因为您试图防止过度拟合。保留正则化技术可以帮助您的模型更好地拟合数据。


# %%
# 重新创建我们在第 1、2、3 和 4 节中构建的数据加载函数。您应该已准备好训练和测试 DataLoader 以供使用。
from torchvision.datasets import ImageFolder
import os
from pathlib import Path
import PIL
from PIL import Image

root_dir = Path("data/") / "pizza_steak_sushi"
train_dir = root_dir / "train"
test_dir = root_dir / "test"

from torchvision import transforms

train_transform = transforms.Compose([
    transforms.RandomRotation(15),             # 随机旋转 ±15 度
    transforms.RandomHorizontalFlip(),         # 随机水平翻转
    transforms.RandomResizedCrop(224),         # 随机裁剪并调整大小到 224x224
    transforms.ColorJitter(brightness=0.2,      # 随机改变亮度
                           contrast=0.2,        # 随机改变对比度
                           saturation=0.2,      # 随机改变饱和度
                           hue=0.1),            # 随机改变色调
    transforms.ToTensor()                      # 转换为张量
])

# 根据root 目录返回dataset
def path_to_dataset(train_dir,test_dir):
    train_dataset = ImageFolder(root=train_dir,
                                )
    