# %%
'''
torchvision 包含通常用于计算机视觉问题的数据集。模型架构和图像转换
torchvision.datasets 计算机视觉数据集示例
torchvision.models 计算机视觉模型架构
torchvision.transforms 图像转换，处理，增强
torch.utils.data.Dataset base dataset class for pytorch
torch.utils.data.DataLoader 在数据集上常见python可迭代对象
'''

from ast import mod
import time
from numpy import argmax
import torch
from torch import nn

import torch.optim.sgd
import torchvision
from torchvision import datasets
from torchvision.transforms import ToTensor

import matplotlib.pyplot as plt

print(f"PyTorch version: {torch.__version__}\ntorchvision version: {torchvision.__version__}")

# %%
# 修正后的数据加载
train_data = datasets.FashionMNIST(
    root="data/fashionMNIST",
    train=True,  # 通常训练集应为 train=True
    download=False,  # 设置为 True 以下载数据集
    transform=ToTensor()  # 正确设置 transform 参数
)

test_data = datasets.FashionMNIST(
    root="data/fashionMNIST",
    train=False,
    download=False,
    transform=ToTensor()
)

# %%
# 获取类别名称
class_names = train_data.classes
print(class_names)

# %%
# 定义函数以显示多张图片
def show_images(dataset, num_images=5):
    plt.figure(figsize=(10, 2))  # 设置图像大小
    for i in range(num_images):
        image, label = dataset[i]
        image_np = image.squeeze().numpy()
        plt.subplot(1, num_images, i+1)
        plt.imshow(image_np, cmap='gray')
        plt.title(class_names[label])
        plt.axis('off')
    plt.show()

# 显示前5张训练集中的图片
show_images(train_data, num_images=5)

# %%
# 显示单张图片
image, label = train_data[0]
print(f"Image shape: {image.shape}\nLabel: {label} ({class_names[label]})")
image_np = image.squeeze().numpy()
plt.imshow(image_np, cmap='gray')
plt.title(f"Label: {class_names[label]}")
plt.axis('off')
plt.show()

# %%
# 使用 DataLoader 进行批量处理
from torch.utils.data import DataLoader

BATCH_SIZE= 32

# 定义 DataLoader
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE, shuffle=False)

# 获取一个批次的数据
train_features_batch, train_labels_batch = next(iter(train_loader))
print(f"Batch image shape: {train_features_batch.shape}")
print(f"Batch train_labels_batch: {train_labels_batch}")
# %%
# Show a sample
torch.manual_seed(42)
random_idx = torch.randint(0, len(train_features_batch), size=[1]).item()
img, label = train_features_batch[random_idx], train_labels_batch[random_idx]
plt.imshow(img.squeeze(), cmap="gray")
plt.title(class_names[label])
plt.axis("Off")
print(f"Image size: {img.shape}")
print(f"Label: {label}, label size: {label.shape}")
# %%
flatten_model = nn.Flatten()

x = train_features_batch[0]

output = flatten_model(x)

print(f"{x.shape}")
print(f"{output.shape}")
# %%
from torch import nn
class FashionMNISTModelV0(nn.Module):
    def __init__(self, input_shape: int, hiddent_units: int , output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = input_shape, out_features=hiddent_units),
            nn.Linear(in_features=hiddent_units,out_features=output_shape)
        )
        
    def forward(self, x):
        return self.layer_stack(x)

torch.manual_seed(42)

model_0 = FashionMNISTModelV0(input_shape=784,
                              hiddent_units=10,
                              output_shape=len(class_names)
                              )

model_0 = model_0.to("cpu")



# %%
import requests
from pathlib import Path 

# Download helper functions from Learn PyTorch repo (if not already downloaded)
if Path("helper_functions.py").is_file():
  print("helper_functions.py already exists, skipping download")
else:
  print("Downloading helper_functions.py")
  # Note: you need the "raw" GitHub URL for this to work
  request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
  with open("helper_functions.py", "wb") as f:
    f.write(request.content)
# %%
from helper_functions import accuracy_fn

loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_0.parameters(),
                            lr = 0.1)

from timeit import default_timer as timer 
def print_train_time(start: float, end: float, device: torch.device = None):
    """Prints difference between start and end time.

    Args:
        start (float): Start time of computation (preferred in timeit format). 
        end (float): End time of computation.
        device ([type], optional): Device that compute is running on. Defaults to None.

    Returns:
        float: time between start and end in seconds (higher is longer).
    """
    total_time = end - start
    print(f"Train time on {device}: {total_time:.3f} seconds")
    return total_time


# %%
from tqdm.auto import tqdm

torch.manual_seed(42)
train_time_start_on_cpu = timer()

epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch:{epoch} \n----------")
    train_loss = 0
    for batch,(X,y) in enumerate(train_loader):
        model_0.train()
        y_pred = model_0(X)
        loss = loss_fn(y_pred,y)
        train_loss += loss.item()
        
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        if batch % 400 == 0:
            print(f"Looked at {batch * len(X)}/ {len(train_loader.dataset)}")
    
    train_loss /= len(train_loader)
    
    test_loss, test_acc = 0, 0
    model_0.eval()
    with  torch.inference_mode():
        for X,y in test_loader:
            test_pred = model_0(X)
            test_loss += loss_fn(test_pred,y).item()
            test_acc += accuracy_fn(y_true=y,y_pred=test_pred.argmax(dim=1))
            
        test_loss /= len(test_loader)
        test_acc /= len(test_loader)
    
    print(f"\nTrain loss:{train_loss:.5f} | Test loss:{test_loss} , Test acc:{test_acc:.2f}%\n")

train_time_end_on_cpu = timer()
total_train_time_model_0 = print_train_time(
    start=train_time_start_on_cpu,
    end=train_time_end_on_cpu,
    device=str(next(model_0.parameters()).device)
)
# %%
torch.manual_seed(42)
def eval_model(model:torch.nn.Module,
               data_loader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               accuracy_fn,
               device:torch.device
               ):
    loss, acc = 0,0
    model.to(device)
    model.eval()
    with torch.inference_mode():
        for X,y in data_loader:
            X,y = X.to(device), y.to(device)
            y_pred = model(X)
            loss += loss_fn(y_pred,y)
            acc += accuracy_fn(
                y_true=y,
                y_pred=y_pred.argmax(dim=1)
            )
        loss /= len(data_loader)
        acc /= len(data_loader)
    return {"model_name": model.__class__.__name__,
            "model_loss":loss.item(),
            "model_acc":acc}

model_0_results = eval_model(model=model_0, data_loader=test_loader,loss_fn=loss_fn,
                             accuracy_fn=accuracy_fn)
model_0_results
# %%
import torch 
device = "mps" if torch.backends.mps.is_available() else "cpu"
device
# %%
class FashionMNISTModelV1(nn.Module):
    def __init__(self, input_shape:int, hidden_units:int,output_shape:int):
        super().__init__()
        self.layer_stack = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features=input_shape,out_features=hidden_units),
            nn.ReLU(),
            nn.Linear(in_features=hidden_units,out_features=output_shape),
            nn.ReLU()
        )
        
    def forward(self, x:torch.Tensor):
        return self.layer_stack(x)

torch.manual_seed(42)
model_1 = FashionMNISTModelV1(
    input_shape=784,
    hidden_units=10,
    output_shape=len(class_names)
).to(device)

next(model_1.parameters()).device

# torch.manual_seed(42)

# %%
from helper_functions import accuracy_fn
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(
    params=model_1.parameters(),
    lr=0.1
)

def train_step(model:torch.nn.Module,
               data_loader:torch.utils.data.DataLoader,
               loss_fn:torch.nn.Module,
               optimizer:torch.optim.Optimizer,
               accuracy_fn,
               device:torch.device=device):
    train_loss,train_acc = 0,0 
    model.to(device)
    for batch,(X,y) in enumerate(data_loader):
        X,y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        train_loss += loss
        train_acc += accuracy_fn(
            y_true=y,
            y_pred=y_pred.argmax(dim=1)
        )
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        # print(f"batch:{batch}")
    
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"Train loss:{train_loss:.5f} | Train accuracy:{train_acc:.2f}%")

def test_step(
    model:torch.nn.Module,
    data_loader:torch.utils.data.DataLoader,
    loss_fn:torch.nn.Module,
    acc_fn,
    device:torch.device=device
):
    model.to(device)
    model.eval()
    with torch.inference_mode():
        loss , acc = 0,0 
        for X,y in data_loader:
            X,y = X.to(device), y.to(device)
            test_pred = model(X)
            loss += loss_fn(test_pred, y)
            acc += acc_fn(y_true=y,
                          y_pred=test_pred.argmax(dim=1))
            # optimizer.zero_grad()
            # loss.backward()
            # optimizer.step()
        
        loss /= len(data_loader)
        acc /= len(data_loader)
        print(f"test loss:{loss:.5f} | test acc:{acc:.2f}%")
        
# %%
torch.manual_seed(42)

from timeit import default_timer as timer
train_time_start_on_gpu = timer()

epochs= 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n----------")
    train_step(
        data_loader=train_loader,
        model=model_1,
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn
    )
    test_step(
        data_loader=test_loader,
        model=model_1,
        loss_fn=loss_fn,
        acc_fn=accuracy_fn
    )
    train_time_end_on_gpu = timer()
    total_train_time_model_1 = print_train_time(
        start=train_time_start_on_gpu,
        end = train_time_end_on_gpu,
        device=device
    )

# %%
torch.manual_seed(42)

# Note: This will error due to `eval_model()` not using device agnostic code 
model_1_results = eval_model(model=model_1, 
    data_loader=test_loader,
    loss_fn=loss_fn, 
    accuracy_fn=accuracy_fn,
    device=device) 
model_1_results 
# %%
# Create a convolutional neural network 
class FashionMNISTModelV2(nn.Module):
    """
    Model architecture copying TinyVGG from: 
    https://poloclub.github.io/cnn-explainer/
    """
    def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(in_channels=input_shape, 
                      out_channels=hidden_units, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=hidden_units, 
                      out_channels=hidden_units,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(hidden_units, hidden_units, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # Where did this in_features shape come from? 
            # It's because each layer of our network compresses and changes the shape of our input data.
            nn.Linear(in_features=hidden_units*7*7, 
                      out_features=output_shape)
        )
    
    def forward(self, x: torch.Tensor):
        x = self.block_1(x)
        # print(x.shape)
        x = self.block_2(x)
        # print(x.shape)
        x = self.classifier(x)
        # print(x.shape)
        return x

torch.manual_seed(42)
model_2 = FashionMNISTModelV2(input_shape=1, 
    hidden_units=10, 
    output_shape=len(class_names)).to(device)
model_2
# %%
torch.manual_seed(42)

# Create sample batch of random numbers with same size as image batch
images = torch.randn(size=(32, 3, 64, 64)) # [batch_size, color_channels, height, width]
test_image = images[0] # get a single image for testing
print(f"Image batch shape: {images.shape} -> [batch_size, color_channels, height, width]")
print(f"Single image shape: {test_image.shape} -> [color_channels, height, width]") 
print(f"Single image pixel values:\n{test_image}")
# %%
torch.manual_seed(42)

# Create a convolutional layer with same dimensions as TinyVGG 
# (try changing any of the parameters and see what happens)
conv_layer = nn.Conv2d(in_channels=3,
                       out_channels=10,
                       kernel_size=3,
                       stride=1,
                       padding=0) # also try using "valid" or "same" here 

# Pass the data through the convolutional layer
conv_layer(test_image) # Note: If running PyTorch <1.11.0, this will error because of shape issues (nn.Conv.2d() expects a 4d tensor as input) 
# %%
# Add extra dimension to test image
test_image.unsqueeze(dim=0).shape
# %%
# Pass test image with extra dimension through conv_layer
conv_layer(test_image.unsqueeze(dim=0)).shape
# %%
torch.manual_seed(42)
# Create a new conv_layer with different values (try setting these to whatever you like)
conv_layer_2 = nn.Conv2d(in_channels=3, # same number of color channels as our input image
                         out_channels=10,
                         kernel_size=(5, 5), # kernel is usually a square so a tuple also works
                         stride=2,
                         padding=0)

# Pass single image through new conv_layer_2 (this calls nn.Conv2d()'s forward() method on the input)
conv_layer_2(test_image.unsqueeze(dim=0)).shape
# %%
# Check out the conv_layer_2 internal parameters
print(conv_layer_2.state_dict())
# %%
# Setup loss and optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(params=model_2.parameters(), 
                             lr=0.1)
# %%
torch.manual_seed(42)

# Measure time
from timeit import default_timer as timer
train_time_start_model_2 = timer()

# Train and test model 
epochs = 3
for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(data_loader=train_loader, 
        model=model_2, 
        loss_fn=loss_fn,
        optimizer=optimizer,
        accuracy_fn=accuracy_fn,
        device=device
    )
    test_step(data_loader=test_loader,
        model=model_2,
        loss_fn=loss_fn,
        acc_fn=accuracy_fn,
        device=device
    )

train_time_end_model_2 = timer()
total_train_time_model_2 = print_train_time(start=train_time_start_model_2,
                                           end=train_time_end_model_2,
                                           device=device)
# %%
# Get model_2 results 
model_2_results = eval_model(
    model=model_2,
    data_loader=test_loader,
    loss_fn=loss_fn,
    accuracy_fn=accuracy_fn
)
model_2_results
# %%
import pandas as pd
compare_results = pd.DataFrame([model_0_results, model_1_results, model_2_results])
compare_results
# %%
# Add training times to results comparison
compare_results["training_time"] = [total_train_time_model_0,
                                    total_train_time_model_1,
                                    total_train_time_model_2]
compare_results
# %%
# Visualize our model results
compare_results.set_index("model_name")["model_acc"].plot(kind="barh")
plt.xlabel("accuracy (%)")
plt.ylabel("model");
# %%
def make_predictions(model: torch.nn.Module, data: list, device: torch.device = device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            # Prepare sample
            sample = torch.unsqueeze(sample, dim=0).to(device) # Add an extra dimension and send sample to device

            # Forward pass (model outputs raw logit)
            pred_logit = model(sample)

            # Get prediction probability (logit -> prediction probability)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 1, so can perform on dim=0)

            # Get pred_prob off GPU for further calculations
            pred_probs.append(pred_prob.cpu())
            
    # Stack the pred_probs to turn list into a tensor
    return torch.stack(pred_probs)
# %%
import random
random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_data), k=9):
    test_samples.append(sample)
    test_labels.append(label)

# View the first test sample shape and label
print(f"Test sample image shape: {test_samples[0].shape}\nTest sample label: {test_labels[0]} ({class_names[test_labels[0]]})")
# %%
# Make predictions on test samples with model 2
pred_probs= make_predictions(model=model_2, 
                             data=test_samples)

# View first two prediction probabilities list
pred_probs[:2]
# %%
# Turn the prediction probabilities into prediction labels by taking the argmax()
pred_classes = pred_probs.argmax(dim=1)
pred_classes
# %%
# Are our predictions in the same form as our test labels? 
test_labels, pred_classes
# %%
# Plot predictions
plt.figure(figsize=(9, 9))
nrows = 3
ncols = 3
for i, sample in enumerate(test_samples):
  # Create a subplot
  plt.subplot(nrows, ncols, i+1)

  # Plot the target image
  plt.imshow(sample.squeeze(), cmap="gray")

  # Find the prediction label (in text form, e.g. "Sandal")
  pred_label = class_names[pred_classes[i]]

  # Get the truth label (in text form, e.g. "T-shirt")
  truth_label = class_names[test_labels[i]] 

  # Create the title text of the plot
  title_text = f"Pred: {pred_label} | Truth: {truth_label}"
  
  # Check for equality and change title colour accordingly
  if pred_label == truth_label:
      plt.title(title_text, fontsize=10, c="g") # green text if correct
  else:
      plt.title(title_text, fontsize=10, c="r") # red text if wrong
  plt.axis(False);
# %%
# Import tqdm for progress bar
from tqdm.auto import tqdm

# 1. Make predictions with trained model
y_preds = []
model_2.eval()
with torch.inference_mode():
  for X, y in tqdm(test_loader, desc="Making predictions"):
    # Send data and targets to target device
    X, y = X.to(device), y.to(device)
    # Do the forward pass
    y_logit = model_2(X)
    # Turn predictions from logits -> prediction probabilities -> predictions labels
    y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1) # note: perform softmax on the "logits" dimension, not "batch" dimension (in this case we have a batch size of 32, so can perform on dim=1)
    # Put predictions on CPU for evaluation
    y_preds.append(y_pred.cpu())
# Concatenate list of predictions into a tensor
y_pred_tensor = torch.cat(y_preds)
# %%
# See if torchmetrics exists, if not, install it
try:
    import torchmetrics, mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")
    assert int(mlxtend.__version__.split(".")[1]) >= 19, "mlxtend verison should be 0.19.0 or higher"
except:
    !pip install -q torchmetrics -U mlxtend # <- Note: If you're using Google Colab, this may require restarting the runtime
    import torchmetrics, mlxtend
    print(f"mlxtend version: {mlxtend.__version__}")
# %%
# Import mlxtend upgraded version
import mlxtend 
print(mlxtend.__version__)
assert int(mlxtend.__version__.split(".")[1]) >= 19 # should be version 0.19.0 or higher
# %%
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# 2. Setup confusion matrix instance and compare predictions to targets
confmat = ConfusionMatrix(num_classes=len(class_names), task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_data.targets)

# 3. Plot the confusion matrix
fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(), # matplotlib likes working with NumPy 
    class_names=class_names, # turn the row and column labels into class names
    figsize=(10, 7)
);
# %%
from pathlib import Path

# Create models directory (if it doesn't already exist), see: https://docs.python.org/3/library/pathlib.html#pathlib.Path.mkdir
MODEL_PATH = Path("models")
MODEL_PATH.mkdir(parents=True, # create parent directories if needed
                 exist_ok=True # if models directory already exists, don't error
)

# Create model save path
MODEL_NAME = "03_pytorch_computer_vision_model_2.pth"
MODEL_SAVE_PATH = MODEL_PATH / MODEL_NAME

# Save the model state dict
print(f"Saving model to: {MODEL_SAVE_PATH}")
torch.save(obj=model_2.state_dict(), # only saving the state_dict() only saves the learned parameters
           f=MODEL_SAVE_PATH)
# %%
# Create a new instance of FashionMNISTModelV2 (the same class as our saved state_dict())
# Note: loading model will error if the shapes here aren't the same as the saved version
loaded_model_2 = FashionMNISTModelV2(input_shape=1, 
                                    hidden_units=10, # try changing this to 128 and seeing what happens 
                                    output_shape=10) 

# Load in the saved state_dict()
loaded_model_2.load_state_dict(torch.load(f=MODEL_SAVE_PATH))

# Send model to GPU
loaded_model_2 = loaded_model_2.to(device)
# %%
# Evaluate loaded model
torch.manual_seed(42)

loaded_model_2_results = eval_model(
    model=loaded_model_2,
    data_loader=test_loader,
    loss_fn=loss_fn, 
    accuracy_fn=accuracy_fn
)

loaded_model_2_results
# %%
model_2_results
# %%
# Check to see if results are close to each other (if they are very far away, there may be an error)
torch.isclose(torch.tensor(model_2_results["model_loss"]), 
              torch.tensor(loaded_model_2_results["model_loss"]),
              atol=1e-08, # absolute tolerance
              rtol=0.0001) # relative tolerance
# %%
# Exercises 练习
# 目前使用计算机视觉的3个工业领域是？
# 1.钢材缺陷检测，车牌识别，人脸识别
# 在机器学习中的过拟合是啥？
# 1.过拟合是指模型在训练期间变现过好，在测试期间表现差。
# 2.模型泛化性能表现差
# 3.主要有多方面引起：数据过大，模型学习到了数据的噪声，模型过大等
# 4.解决方案：特征降维，模型简化，加入正则化，dropout，剪枝，早停法
# 加载 torchvision.datasets.MNIST() 训练和测试数据集
from torchvision import datasets
from torchvision.transforms import ToTensor

train_datasets = datasets.MNIST(
    root="data",
    train=True,
    download=True,
    transform=ToTensor()
)

test_datasets = datasets.MNIST(
    root="data",
    train=False,
    download=True,
    transform=ToTensor()
)

class_names = train_datasets.classes
print(class_names)
# %%
# 可视化 MNIST 训练数据集的至少 5 个不同样本。
import matplotlib.pyplot as plt

torch.manual_seed(42)

def img_show(dataset,num=5):
    plt.figure(figsize=(10,2))
    for i in range(num):
        img,label = dataset[i]
        img_np = img.squeeze().numpy()
        plt.subplot(1,num,i+1)
        plt.imshow(img_np)
        plt.title(class_names[label])
        plt.axis('off')
    plt.show()
    
img_show(train_datasets,5)
# %%
from torch.utils.data import DataLoader
BATCH_SIZE = 32
train_dataloader = DataLoader(train_datasets,BATCH_SIZE,
                              shuffle=True)
test_dataloader = DataLoader(test_datasets,batch_size=BATCH_SIZE,
                             shuffle=False)

# %%
train_batch_img,train_batch_label = next(iter(train_dataloader))
print(f"train_batch_img.shape:{train_batch_img.shape}")
print(f"train_batch_label.shape:{train_batch_label.shape}")

# %%
import torch.nn as nn
class MyModel(nn.Module):
    def __init__(self, input_shape,hidden_shape,output_shape):
        super().__init__()
        self.block_1 = nn.Sequential(
            nn.Conv2d(input_shape,hidden_shape,(3,3),
                      stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(hidden_shape,hidden_shape,kernel_size=(3,3),
                      stride=(1,1),padding=(1,1)),
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,
                         ceil_mode=False)
        )
        self.block_2 = nn.Sequential(
            nn.Conv2d(hidden_shape,hidden_shape,(3,3),
                      stride=(1,1),padding=(1,1)),
            nn.ReLU(),
            nn.Conv2d(hidden_shape,hidden_shape,kernel_size=(3,3),
                      stride=(1,1),padding=(1,1)),
            nn.ReLU(),            
            nn.MaxPool2d(kernel_size=2,stride=2,padding=0,dilation=1,
                         ceil_mode=False)
        )
        self.classifier=nn.Sequential(
            nn.Flatten(1,-1),
            nn.Linear(in_features=hidden_shape * 7 * 7 ,out_features=output_shape,bias=True)
        ) 
    
    def forward(self, X):
        block_1 = self.block_1(X)       
        block_2 = self.block_2(block_1)
        return self.classifier(block_2)
    
mymodel = MyModel(input_shape=1,hidden_shape=10,output_shape=10)
print(mymodel)


# %%
from helper_functions import accuracy_fn
cpu_device = "cpu"
mps_device = "mps"

# epochs = 100
loss_fn = nn.CrossEntropyLoss()
acc_fn = accuracy_fn
optimizer = torch.optim.SGD(mymodel.parameters(),lr=0.1)
def train_step(model:nn.Module, data_loader:torch.utils.data.DataLoader,
               optimizer:torch.optim.Optimizer,
               loss_fn:nn.Module,
               acc_fn,
               device:torch.device):
    # set device
    model.to(device)
    train_loss , train_acc = 0,0
    model.train()
    for batch,(X,y) in enumerate(data_loader):
        X,y = X.to(device),y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred,y)
        train_loss += loss
        train_acc += acc_fn(y,y_pred.argmax(dim=1))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(data_loader)
    train_acc /= len(data_loader)
    print(f"train_loss:{train_loss:.5f} train_acc:{train_acc:.2f}% ")

def test_step(model:nn.Module,data_loader:torch.utils.data.DataLoader,
              loss_fn:nn.Module,acc_fn,device:torch.device):
    model.to(device)
    model.eval()
    test_loss,test_acc = 0,0
    with torch.inference_mode():
        for X,y in data_loader:
            X,y = X.to(device),y.to(device)
            test_pred = model(X)
            loss = loss_fn(test_pred,y)
            test_loss += loss
            test_acc += acc_fn(y,test_pred.argmax(dim=1))
        test_loss /= len(data_loader)
        test_acc /= len(data_loader)
        print(f"test_loss:{test_loss:.5f} | test_acc:{test_acc:.2f}%")


# %%
from timeit import default_timer as timer
from tqdm.auto import tqdm

torch.manual_seed(42)
train_start_cpu_on_timer = timer()
epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(mymodel,train_dataloader,optimizer,
           loss_fn=loss_fn,acc_fn=acc_fn,device=cpu_device)
    test_step(model=mymodel,data_loader=test_dataloader,loss_fn=loss_fn,
              acc_fn=acc_fn,device=cpu_device)

train_end_cpu_on_timer = timer()

total_train_time_model_2 = print_train_time(start=train_start_cpu_on_timer,
                                           end=train_end_cpu_on_timer,
                                           device=cpu_device)
# %%
torch.manual_seed(42)
train_start_cpu_on_timer = timer()
epochs = 3

for epoch in tqdm(range(epochs)):
    print(f"Epoch: {epoch}\n---------")
    train_step(mymodel,train_dataloader,optimizer,
           loss_fn=loss_fn,acc_fn=acc_fn,device=mps_device)
    test_step(model=mymodel,data_loader=test_dataloader,loss_fn=loss_fn,
              acc_fn=acc_fn,device=mps_device)

train_end_cpu_on_timer = timer()

total_train_time_model_2 = print_train_time(start=train_start_cpu_on_timer,
                                           end=train_end_cpu_on_timer,
                                           device=mps_device)
# %%
import torch

cpu_device = torch.device("cpu")
mps_device = torch.device("mps") if torch.backends.mps.is_available() else torch.device("cpu")
def make_predictions(model:torch.nn.Module, data:list, device:torch.device):
    pred_probs = []
    model.eval()
    with torch.inference_mode():
        for sample in data:
            sample = torch.unsqueeze(sample, dim=0).to(device)
            pred_logit = model(sample)
            pred_prob = torch.softmax(pred_logit.squeeze(), dim=0)
            pred_probs.append(pred_prob.cpu())
        return torch.stack(pred_probs)
    
import random
random.seed(42)
test_samples = []
test_labels = []
for sample, label in random.sample(list(test_datasets), k=9):
    test_samples.append(sample)
    test_labels.append(label)
pred_probs = make_predictions(model=mymodel,
                              data=test_samples,
                              device=mps_device)
pred_probs[:2]
# %%
pred_classes = pred_probs.argmax(dim=1)
pred_classes
# %%
plt.figure(figsize=(9,9))

nrows = 3
ncols = 3

for i, sample in enumerate(test_samples):
    plt.subplot(nrows, ncols, i+1)
    plt.imshow(sample.squeeze(), cmap='gray')
    pred_label = class_names[pred_classes[i]]
    truth_label = class_names[test_labels[i]]
    title_text = f"Pred:{pred_label} | Truth:{truth_label}"
    if pred_label == truth_label:
        plt.title(title_text, fontsize=10, c='g')
    else:
        plt.title(title_text, fontsize=10, c='r')
    plt.axis(False)
# %%
from tqdm.auto import tqdm

y_preds = []
mymodel.to(cpu_device)
mymodel.eval()
with torch.inference_mode():
    for X,y in tqdm(test_dataloader, desc='Making prediction'):
        X,y =  X.to(cpu_device), y.to(cpu_device)
        y_logit = mymodel(X)
        y_pred = torch.softmax(y_logit, dim=1).argmax(dim=1)
        y_preds.append(y_pred.cpu())
        
y_pred_tensor = torch.cat(y_preds)

# 绘制一个混淆矩阵，将模型的预测与真值标签进行比较。
from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

confmat = ConfusionMatrix(num_classes=len(class_names),
                          task='multiclass')
confmat_tensor = confmat(preds=y_pred_tensor,
                         target=test_datasets.targets.to(cpu_device))

fig, ax = plot_confusion_matrix(
    conf_mat=confmat_tensor.numpy(),
    class_names = class_names,
    figsize=(10,7)
)

# %%
a = torch.rand((1,3,64,64))
a
# %%
a_c = nn.Conv2d(3,10,3)(a)
a_c.shape
# %%
a_c_1 = nn.Conv2d(3,10,5)(a)
a_c_1.shape
# %%
from torchsummary import summary

# 假设你的模型已经实例化为 mymodel
summary(mymodel, (1, 28, 28))  # 这里的 (1, 28, 28) 是输入数据的形状，根据你的实际情况进行修改
# %%
