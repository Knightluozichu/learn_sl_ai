# %%
# 检查版本
try:
    import torch
    import torchvision
    assert int(torch.__version__.split(".")[0]) >= 2 or int(torch.__version__.split(".")[1]) >= 12 , "torch version >= 1.2.0+"
    assert int(torchvision.__version__.split(".")[1]) >= 13, "torchvision version >= 0.13+"
except (ImportError, AssertionError) as e:
    print(f"[INFO] torch/torchvision version not as required, installing nightly versions.")
    !pip3 install -U torch torchvision torchaudio --extra-index-url https://download.pytorch.org/whl/cu113
    import torch
    import torchvision
    print(f"torch version: {torch.__version__}")
    print(f"torchvision version: {torchvision.__version__}")

    
# %%
# 编写与设备无关的代码
from calendar import c
import matplotlib.pyplot as plt

from regex import T
import torch
import torchinfo
import torchvision

from torch import classes, nn
from torchvision import transforms
from zmq import device

try:
    from torchinfo import summary
except:
    print(f"[INFO] Couldn't find torchinfo... installing it.")
    !pip install -q torchinfo
    from torchinfo import summary
    
try:
    from going_modular import data_setup, engine
    from helper_functions import download_data, set_seeds,plot_loss_curves
except:
    print("[INFO] Couldn't find going_modular or helper_functions... downloading them from GitHub.")
    !git clone https://github.com/mrdbourke/pytorch-deep-learning
    !mv pytorch-deep-learning/going_modular .
    !mv pytorch-deep-learning/helper_functions.py . # get the helper_functions.py script
    !rm -rf pytorch-deep-learning
    from going_modular import data_setup, engine
    from helper_functions import download_data, set_seeds, plot_loss_curves
    
device = torch.device("mps") if torch.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
# torch.set_default_device(device)
device


# %%
# 获取数据
# Download pizza, steak, sushi images from GitHub
data_20_percent_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi_20_percent.zip",
                                     destination="pizza_steak_sushi_20_percent")

data_20_percent_path
# %%
train_dir = data_20_percent_path / "train"
test_dir = data_20_percent_path / "test"

# %%
# 获取预训练模型权重，数据增强处理自己数据，使用权重创建预训练模型，冻结权重，更改输出头

effnetb2_weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
efffnetb2_transforms = effnetb2_weights.transforms()

effnetb2 = torchvision.models.efficientnet_b2(weights=effnetb2_weights)

for param in effnetb2.parameters():
    param.requires_grad = False
    
# %%
effnetb2.classifier
# summary(effnetb2, input_size=(1, 3, 300, 300), device=device)
# %%
# 更改输出头
effnetb2.classifier = nn.Sequential(
    nn.Dropout(0.3, inplace=True),
    nn.Linear(1408, 3)
)
# %%
# 创建函数来制作 EffNetB2 特征提取器
def create_effnetb2_model(num_classes:int=3,
                          seed:int=42):
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    b2_weights= torchvision.models.EfficientNet_B2_Weights.DEFAULT
    
    b2_model = torchvision.models.efficientnet_b2(weights=b2_weights)
    
    b2_transforms = b2_weights.transforms()
    
    for param in b2_model.parameters():
        param.requires_grad = False
        
    b2_model.classifier = nn.Sequential(
        nn.Dropout(0.3, inplace=True),
        nn.Linear(1408, num_classes)
    )
    
    return b2_model, b2_transforms
    
    
# %%
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=3)

# %%
from torchinfo import summary

summary(effnetb2,
        input_size=(1,3,224,224),
        col_names=["input_size", "output_size", "num_params", "kernel_size", "trainable"],
        col_width=20,
        row_settings=["var_names"])

# %%
# 我们的 EffNetB2 特征提取器已准备就绪，是时候创建一些 DataLoader了

from going_modular import data_setup

train_dataloader_effnetb2, test_dataloader_effnetb2, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    batch_size=32,
    transform=effnetb2_transforms,
    num_workers=0
)


# 训练 EffNetB2 特征提取器
from going_modular import engine

optimizer = torch.optim.Adam(params=effnetb2.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

set_seeds(42)
effnetb2_results = engine.train(model=effnetb2,
                                        optimizer=optimizer,
                                        loss_fn=loss_fn,
                                        train_dataloader=train_dataloader_effnetb2,
                                        test_dataloader=test_dataloader_effnetb2,
                                        epochs=10,
                                        device=device)

# %%
from helper_functions import plot_loss_curves

plot_loss_curves(effnetb2_results)
# %%
# 保存模型

from going_modular import utils

utils.save_model(model=effnetb2,
                 target_dir="models",
                 model_name="09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth")

# %% 检查模型大小
from pathlib import Path

pretrained_effnetb2_model_size = Path("models/09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth").stat().st_size

print(f"Pretrained EffNetB2 feature extractor model size:{pretrained_effnetb2_model_size/1e6} MB")

# %%
effnetb2_total_params = sum(p.numel() for p in effnetb2.parameters())
effnetb2_total_params
# %%
effnetb2_stats = {"test_loss":effnetb2_results["test_loss"][-1],
                 "test_acc":effnetb2_results["test_acc"][-1],
                 "total_params":effnetb2_total_params,
                 "model_size(MB)":pretrained_effnetb2_model_size/1e6}
effnetb2_stats
# %% 创建 ViT 特征提取器

vit = torchvision.models.vit_b_16()
vit.heads
# %%
def create_vit_model(num_classes:int=3,
                     seed:int=42):
    torch.manual_seed(seed)
    torch.mps.manual_seed(seed)
    
    weights = torchvision.models.ViT_B_16_Weights.DEFAULT
    transforms = weights.transforms()
    model=torchvision.models.vit_b_16(weights=weights)
    
    for param in model.parameters():
        param.requires_grad = False
    
    model.heads = nn.Sequential(
        nn.Linear(in_features=768, out_features=num_classes)
    )
    
    return model, transforms
# %%
vit, vit_transforms = create_vit_model(num_classes=3)
# %%
summary(vit,
        input_size=(1,3,224,224),
        col_names=["input_size", "output_size", "num_params", "kernel_size", "trainable"],
        col_width=20,
        row_settings=["var_names"])
# %%
from going_modular import data_setup

train_dataloader_vit, test_dataloader_vit, class_names = data_setup.create_dataloaders(
    train_dir=train_dir,
    test_dir=test_dir,
    batch_size=32,
    transform=vit_transforms,
)

# %%
from going_modular import engine

optimizer = torch.optim.Adam(params=vit.parameters(), lr=1e-3)
loss_fn = torch.nn.CrossEntropyLoss()

set_seeds(42)
vit_results = engine.train(model=vit,
                           train_dataloader=train_dataloader_vit,
                            test_dataloader=test_dataloader_vit,
                            optimizer=optimizer,
                            loss_fn=loss_fn,
                            epochs=10,
                            device=device)

# %%
from helper_functions import plot_loss_curves

plot_loss_curves(vit_results)
# %%
from going_modular import utils

utils.save_model(model=vit,
                 target_dir="models",
                 model_name="09_pretrained_vit_feature_extractor_pizza_steak_sushi_20_percent.pth")

# %%
from pathlib import Path

pretrained_vit_model_size = Path("models/09_pretrained_vit_feature_extractor_pizza_steak_sushi_20_percent.pth").stat().st_size / 1e6
print(f"Pretrained ViT feature extractor model size: {pretrained_vit_model_size} MB")
# %%
vit_total_params = sum(p.numel() for p in vit.parameters())
print(f"Total parameters in ViT model: {vit_total_params}")
# %%
vit_stats = {"test_loss":vit_results["test_loss"][-1],
            "test_acc":vit_results["test_acc"][-1],
            "total_params":vit_total_params,
            "model_size(MB)":pretrained_vit_model_size}
vit_stats
# %%
from pathlib import Path
print(f"[INFO] Finding all filepaths ending with '.jpg in {test_dir}")
test_data_paths = list(Path(test_dir).rglob("*.jpg"))
test_data_paths[:5]
# %% 创建函数以在测试数据集中进行预测
import pathlib
import torch

from PIL import Image
from timeit import default_timer as timer
from tqdm.auto import tqdm
from typing import List, Dict

def pred_and_store(paths:List[pathlib.Path],
                   model:torch.nn.Module,
                   transform:torchvision.transforms,
                   class_names:List[str],
                   device:torch.device=device)->List[Dict]:
    pred_list = []
    
    for path in tqdm(paths):
        pred_dict = {}
        
        pred_dict["image_path"] = path
        class_name = path.parent.stem
        pred_dict["class_name"] = class_name
        
        start_time = timer()
        img = Image.open(path)
        
        transformed_image = transform(img).unsqueeze(0).to(device)
        model.to(device)
        model.eval()
        
        with torch.inference_mode():
            pred_logit = model(transformed_image)
            pred_prob = torch.softmax(pred_logit, dim=1)
            pred_label = torch.argmax(pred_prob, dim=1)
            pred_class = class_names[pred_label.cpu()]
            
            pred_dict["pred_prob"] = round(pred_prob.unsqueeze(0).max().item(), 4)
            pred_dict["pred_class"]  = pred_class
            end_time = timer()
            pred_dict["time_for_pred"] = round(end_time - start_time, 4)
            
        pred_dict["correct"] = class_name == pred_class
        pred_list.append(pred_dict)
        
    return pred_list


        


# %%
effnetb2_test_pred_dicts = pred_and_store(paths=test_data_paths,
                                          model=effnetb2,
                                          transform=effnetb2_transforms,
                                          class_names=class_names,
                                          device="cpu")
# %%
effnetb2_test_pred_dicts[:2]
# %%
import pandas as pd

effnetb2_test_pred_df = pd.DataFrame(effnetb2_test_pred_dicts)
effnetb2_test_pred_df.head()

# %%
effnetb2_test_pred_df.correct.value_counts()
# %%
effnetb2_average_time_per_pred = round(effnetb2_test_pred_df.time_for_pred.mean(),4)
print(f"EffNetB2 average time per prediction: {effnetb2_average_time_per_pred} seconds")
# %%
vit_test_pred_dicts = pred_and_store(paths=test_data_paths,
                                     model=vit,
                                     transform=vit_transforms,
                                     class_names=class_names,
                                     device="cpu")

# %%
vit_test_pred_dicts[:2]
# %%
vit_test_pred_df = pd.DataFrame(vit_test_pred_dicts)
vit_test_pred_df.head()
# %%
vit_test_pred_df.correct.value_counts()
# %%
vit_average_time_per_pred = round(vit_test_pred_df.time_for_pred.mean(),4)
# %%
print(f"ViT average time per prediction: {vit_average_time_per_pred} seconds")
# %%
vit_stats["time_per_pred_cpu"] = vit_average_time_per_pred
vit_stats

# %%
effnetb2_stats["time_per_pred_cpu"] = effnetb2_average_time_per_pred
effnetb2_stats
# %%
df = pd.DataFrame([effnetb2_stats, vit_stats])

df["model"] = ["EffNetB2", "ViT"]
df["test_acc"] = df["test_acc"].apply(lambda x: round(x * 100, 2))
# %%
df
# %%
pd.DataFrame(data=(df.set_index("model").loc["ViT"] / df.set_index("model").loc["EffNetB2"]),
             columns=["ViT to EffNetB2 ratios"]).T
# %%
fig, ax = plt.subplots(figsize=(12, 8))
scatter = ax.scatter(data=df,
                     x="time_per_pred_cpu",
                     y="test_acc",
                     c=["blue","orange"],
                     s=df["model_size(MB)"])
ax.set_title("FoodVision Mini Inference Speed vs Performance", fontsize=18)
ax.set_xlabel("Prediction time per image (seconds)", fontsize=14)
ax.set_ylabel("Test accuracy(%)", fontsize=14)
ax.tick_params(axis="both", labelsize=12)
ax.grid(True)

for index, row in df.iterrows():
    ax.annotate(text=row["model"],
                xy=(row["time_per_pred_cpu"] + 0.0006, row["test_acc"] + 0.03),
                size=12)
handles, labels = scatter.legend_elements(prop="sizes",alpha=0.5)
model_size_legend = ax.legend(handles,
                              labels,
                              loc="upper right",
                              title="Model size (MB)",
                              fontsize=12)
!mkdir images/
plt.savefig("images/09-foodvision-mini-inference-speed-vs-performance.jpg")
plt.show()
# %%
try:
    import gradio as gr
except ImportError:
    !pip install -q gradio
    import gradio as gr

print(f"[INFO] Gradio version: {gr.__version__}")
# %%
# Gradio 的总体前提与我们在整个课程中重复的内容非常相似。
# 我们的输入和输出是什么？
# 我们应该如何到达那里？
# 这就是我们的机器学习模型的作用。
# inputs -> ML model -> outputs
# 在我们的例子中，对于 FoodVision Mini，我们的输入是食物的图像，我们的 ML 模型是 EffNetB2，我们的输出是食物类别（比萨饼、牛排或寿司）。
# images of food -> EffNetB2 -> outputs
# 您的输入和输出可能是以下各项的任意组合：

# Images 图像
# Text 发短信
# Video 视频
# Tabular data 表格数据
# Audio 音频
# Numbers 数字
# & more & 更多

# with torch.device("cpu"):
#     print(next(iter(effnetb2.parameters())).device)
effnetb2.to("cpu")
next(iter(effnetb2.parameters())).device
# %%
from typing import Tuple, Dict

def predict(img) ->Tuple[Dict, float]:
    start_time = timer()
    img = effnetb2_transforms(img).unsqueeze(0).to("cpu")
    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    pred_time = round(timer() - start_time, 5)
    return pred_labels_and_probs, pred_time


# %%
import random 
from PIL import Image

test_data_paths = list(Path(test_dir).rglob("*.jpg"))

random_image_path = random.sample(test_data_paths,k=1)[0]

image = Image.open(random_image_path)

print(f"[INFO] Predicting on image at path: {random_image_path}")

pred_dict, pred_time = predict(img=image)
print(f"Prediction label and probability dictionary: \n{pred_dict}")
print(f"Prediction time: {pred_time} seconds")

# %%
# Create a list of example inputs to our Gradio demo
example_list = [[str(filepath)] for filepath in random.sample(test_data_paths, k=3)]
example_list
# %%
import gradio as gr

title = "FoodVision Mini - EffNetB2 - 🍕🥩🍣"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
article = "Created at [09. PyTorch Model Deployment](https://www.learnpytorch.io/09_pytorch_model_deployment/)."

demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"),
                             gr.Number(label="Predcition time (seconds)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)
demo.launch(debug=False,
            share=True)

# %%
# 我们已经通过 Gradio 演示看到了我们的 FoodVision Mini 模型栩栩如生。

# But what if we wanted to share it with our friends?
# 但是，如果我们想与朋友分享呢？

# Well, we could use the provided Gradio link, however, the shared link only lasts for 72-hours.
# 好吧，我们可以使用提供的 Gradio 链接，但是，共享链接仅持续 72 小时。

# To make our FoodVision Mini demo more permanent, we can package it into an app and upload it to Hugging Face Spaces.
# 为了使我们的 FoodVision Mini 演示更加持久，我们可以将其打包到一个应用程序中，并将其上传到 Hugging Face Spaces。

# 创建一个 demos 文件夹来存储我们的 FoodVision Mini 应用程序文件
import shutil
from pathlib import Path
foodvision_mini_demo_path = Path("demos/foodvision_mini/")

if foodvision_mini_demo_path.exists():
    shutil.rmtree(foodvision_mini_demo_path)

foodvision_mini_demo_path.mkdir(parents=True,
                                exist_ok=True)

!ls demos/foodvision_mini/

# %%
import shutil
from pathlib import Path

foodvision_mini_examples_path = foodvision_mini_demo_path / "examples"
foodvision_mini_examples_path.mkdir(parents=True,
                                    exist_ok=True)

foodvision_mini_examples = [Path('data/pizza_steak_sushi_20_percent/test/pizza/61656.jpg'),
                            Path('data/pizza_steak_sushi_20_percent/test/steak/48208.jpg'),
                            Path('data/pizza_steak_sushi_20_percent/test/sushi/1063878.jpg')]

for example in foodvision_mini_examples:
    destination = foodvision_mini_examples_path / example.name
    print(f"[INFO] Copying {example} to {destination}")
    shutil.copy2(example, destination)

# %%
import os

example_list = [["examples/" + example] for  example in os.listdir(foodvision_mini_examples_path)]
example_list
# %%
import shutil
effnetb2_foodvision_mini_model_path = "models/09_pretrained_effnetb2_feature_extractor_pizza_steak_sushi_20_percent.pth"
effnetb2_foodvision_mini_model_destination = foodvision_mini_demo_path / effnetb2_foodvision_mini_model_path.split("/")[1]
try:
    print(f"[INFO Attenpting to move {effnetb2_foodvision_mini_model_path} to {effnetb2_foodvision_mini_model_destination}]")
    shutil.move(effnetb2_foodvision_mini_model_path, effnetb2_foodvision_mini_model_destination)
    
    print(f"[INFO] Modell move complete.")
except FileNotFoundError as e:
    print(f"[INFO] No model found at {effnetb2_foodvision_mini_model_path}, perhaps its already been moved?")
    print(f"[INFO] Model exists at {effnetb2_foodvision_mini_model_destination}: {effnetb2_foodvision_mini_model_destination.exists()}")
# %%
# %%writerfile demos/foodvision_mini/model.py
# import torch
# import torchvision

# from torch import nn
# def create_effnetb2_model(num_classes:int=3,
#                           seed:int=42):
#     weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
#     transforms = weights.transforms()
#     model = torchvision.models.efficientnet_b2(weights=weights)
    
#     for param in model.parameters():
#         param.requires_grad = False
    
#     torch.manual_seed(seed)
#     model.classifier = nn.Sequential(
#         nn.Dropout(0.3, inplace=True),
#         nn.Linear(1408, num_classes)
#     )
    
#     return model,transforms

# 替代 %%writerfile 的代码
file_path = "demos/foodvision_mini/model.py"

# 定义内容
file_content = """import torch
import torchvision

from torch import nn
def create_effnetb2_model(num_classes:int=3,
                          seed:int=42):
    weights = torchvision.models.EfficientNet_B2_Weights.DEFAULT
    transforms = weights.transforms()
    model = torchvision.models.efficientnet_b2(weights=weights)
    
    for param in model.parameters():
        param.requires_grad = False
    
    torch.manual_seed(seed)
    model.classifier = nn.Sequential(
        nn.Dropout(0.3, inplace=True),
        nn.Linear(1408, num_classes)
    )
    
    return model,transforms
"""

# 写入文件
with open(file_path, "w") as f:
    f.write(file_content)

print(f"File written to {file_path}")

# %%
file_path = "demos/foodvision_mini/app.py"

file_context = """import gradio as gr
import os
import torch

from model import create_effenetb2_model
from timeit import default_timer as timer
from typing import Tuple, Dict

class_names = ["pizza", "steak", "sushi"]
effnetb2, effnetb2_transforms = create_effnetb2_model(num_classes=len(class_names))
effnetb2.load_state_dict(torch.load(
    f"09_pretrained_effnetb2_feature_extrator_pizza_steak_sushi_20_percent.pth",
    map_location=torch.device("cpu")
))

def predict(img) ->Tuple[Dict, float]:
    start_time =timer()
    img=effnetb2_transforms(img).unsqueeze(0).to("cpu")
    effnetb2.eval()
    with torch.inference_mode():
        pred_probs = torch.softmax(effnetb2(img), dim=1)
    pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
    pred_time = round(timer() - start_time, 5)
    return pred_labels_and_probs, pred_time

title = "迷你食物视觉 - EffNetB2 - 🍕🥩🍣"
description = "一个 EfficientNetB2 特征提取器计算机视觉模型，用于将食物图像分类为比萨饼、牛排或寿司。"
article = "由 子初 创建。"

example_list = [["examples/" + example] for example in os.listdir("examples")]
demo = gr.Interface(fn=predict,
                    inputs=gr.Image(type="pil"),
                    outputs=[gr.Label(num_top_classes=3, label="预测"),
                             gr.Number(label="预测时间 (seconds)")],
                    examples=example_list,
                    title=title,
                    description=description,
                    article=article)

demo.launch()"""

with open(file_path, "w") as f:
    f.write(file_context)

# %%
file_path = "demos/foodvision_mini/requirements.txt"
file_content = """
torch>=1,12,0
torchvision>=0.13.0
gradio>=3.1.4
"""

with open(file_path, "w") as f:
    f.write(file_content)
print(f"File written to {file_path} completed.")
# %%
!ls demos/foodvision_mini
# %%
# !zip -r ../foodvision_mini.zip * -x "*.pyc" "*.ipynb" "*__pycache_*" "*ipynb_checkpoints*"
# %%
# Change into and then zip the foodvision_mini folder but exclude certain files
!cd demos/foodvision_mini && zip -r ../foodvision_mini.zip * -x "*.pyc" "*.ipynb" "*__pycache__*" "*ipynb_checkpoints*"

# Download the zipped FoodVision Mini app (if running in Google Colab)
try:
    from google.colab import files
    files.download("demos/foodvision_mini.zip")
except:
    print("Not running in Google Colab, can't use google.colab.files.download(), please manually download.")

# %%
# IPython is a library to help make Python interactive
from IPython.display import IFrame

# Embed FoodVision Mini Gradio demo
IFrame(src="https://huggingface.co/spaces/rainknightpox/foodvision_mini", width=900, height=750)
# %%
import torch
import torchvision
from torch import nn

effnetb2_food101, effnetb2_transforms = create_effnetb2_model(num_classes=101)

from torchinfo import summary

summary(effnetb2_food101,
        input_size=(1,3,224,224),
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])
# %%
food101_train_transforms = torchvision.transforms.Compose([
    torchvision.transforms.TrivialAugmentWide(),
    effnetb2_transforms,
])
print(f"Training transforms:\n{food101_train_transforms}\n") 
print(f"Testing transforms:\n{effnetb2_transforms}")
# %%
from torchvision import datasets

from pathlib import Path
data_dir = Path("data")

train_data = datasets.Food101(root=data_dir,
                              split="train",
                              transform=food101_train_transforms,
                              download=True)
test_data = datasets.Food101(root=data_dir,
                                split="test",
                                transform=effnetb2_transforms,
                                download=True)

# %%
food101_class_names= train_data.classes
food101_class_names[:10]
# %%
# torch.utils.data.random_split(train_data, [len(train_data) - 1000, 1000])

def split_dataset(dataset:torchvision.datasets, split_size:float=0.2, seed:int=42):
    length_1 = int(len(dataset) * split_size)
    length_2 = len(dataset) - length_1
    
    print(f"[INFO] Splitting dataset of length {len(dataset)} into splits of size: {length_1} ({int(split_size*100)}%), {length_2} ({int((1-split_size)*100)}%)")

    random_split_1, random_split_2 = torch.utils.data.random_split(
        dataset,
        [length_1, length_2],
        generator=torch.Generator().manual_seed(seed)
    )
    
    return random_split_1, random_split_2

train_data_food101_20_percent, _=split_dataset(dataset=train_data,
                                               split_size=0.2)
test_data_food101_20_percent, _=split_dataset(dataset=test_data,
                                              split_size=0.2)
len(train_data_food101_20_percent), len(test_data_food101_20_percent)

# %%
import os
import torch
BATCH_SIZE=32
NUM_WORKERS = 0

train_dataloader_food101_20_percent = torch.utils.data.DataLoader(train_data_food101_20_percent,
                                                                  batch_size=BATCH_SIZE,
                                                                  shuffle=True,
                                                                  num_workers=NUM_WORKERS)
test_dataloader_food101_20_percent = torch.utils.data.DataLoader(test_data_food101_20_percent,
                                                                    batch_size=BATCH_SIZE,
                                                                    shuffle=False,
                                                                    num_workers=NUM_WORKERS)

# %%
from going_modular import engine
from helper_functions import set_seeds

device = torch.device("mps") if torch.mps.is_available() else torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

optimizer =torch.optim.Adam(params=effnetb2_food101.parameters(), 
                            lr=1e-3)

loss_fn = torch.nn.CrossEntropyLoss()

set_seeds()

effnetb2_food101_results = engine.train(model=effnetb2_food101,
                                        train_dataloader=train_dataloader_food101_20_percent,
                                        test_dataloader=test_dataloader_food101_20_percent,
                                        optimizer=optimizer,
                                        loss_fn=loss_fn,
                                        epochs=5,
                                        device=device)
# %%from pathlib import Path

# Create FoodVision Big demo path
foodvision_big_demo_path = Path("demos/foodvision_big/")

# Make FoodVision Big demo directory
foodvision_big_demo_path.mkdir(parents=True, exist_ok=True)

# Make FoodVision Big demo examples directory
(foodvision_big_demo_path / "examples").mkdir(parents=True, exist_ok=True)

foodvision_big_class_names_path = foodvision_big_demo_path / "class_names.txt"

with open(foodvision_big_class_names_path, "w") as f:
    f.write("\n".join(food101_class_names))
    print((f"[INFO] Class names written to {foodvision_big_class_names_path}"))


# Open Food101 class names file and read each line into a list
with open(foodvision_big_class_names_path, "r") as f:
    food101_class_names_loaded = [food.strip() for food in  f.readlines()]
    
# View the first 5 class names loaded back in
food101_class_names_loaded[:5]
# %%
from helper_functions import plot_loss_curves

# Check out the loss curves for FoodVision Big
plot_loss_curves(effnetb2_food101_results)

from going_modular import utils

# Create a model path
effnetb2_food101_model_path = "09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth" 

# Save FoodVision Big model
utils.save_model(model=effnetb2_food101,
                 target_dir="models",
                 model_name=effnetb2_food101_model_path)

# %%
# Create Food101 compatible EffNetB2 instance
loaded_effnetb2_food101, effnetb2_transforms = create_effnetb2_model(num_classes=101)

# Load the saved model's state_dict()
loaded_effnetb2_food101.load_state_dict(torch.load("models/09_pretrained_effnetb2_feature_extractor_food101_20_percent.pth"))

# %%
from pathlib import Path

# Get the model size in bytes then convert to megabytes
pretrained_effnetb2_food101_model_size = Path("models", effnetb2_food101_model_path).stat().st_size // (1024*1024) # division converts bytes to megabytes (roughly) 
print(f"Pretrained EffNetB2 feature extractor Food101 model size: {pretrained_effnetb2_food101_model_size} MB")
# %%
