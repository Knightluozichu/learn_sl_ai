# %%

if "__file__" in globals():
    import os, sys

    sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

# from requests import get
import torch
import torchvision
from torchvision.models.detection import fasterrcnn_resnet50_fpn
from VOC2007_config import CLASS_NAMES, ROOT_DIR
from VOC2007_dataset import get_datasets

# 获取类别数量
num_classes = len(CLASS_NAMES) + 1  # 包括背景类

# 加载预训练的 Faster R-CNN 模型
model = fasterrcnn_resnet50_fpn(pretrained=True)

# 替换分类头
in_features = model.roi_heads.box_predictor.cls_score.in_features
model.roi_heads.box_predictor = (
    torchvision.models.detection.faster_rcnn.FastRCNNPredictor(in_features, num_classes)
)
# 检查是否有gpu可用


# 将模型移动到设备上
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
model.to(device)

# 获取数据集和数据加载器
train_loader, val_loader = get_datasets()

# 定义优化器
params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)
#  %%
# 打印train_loader的前2个数据
for images, targets in train_loader:
    print(images)
    print(targets)
    break

# %%

from tqdm import tqdm
import torch

num_epochs = 10

# 外层进度条显示 epoch
for epoch in tqdm(range(num_epochs), desc='Epochs'):
    model.train()
    total_loss = 0
    # 内层进度条显示每个 epoch 中的 batch 进度
    with tqdm(train_loader, desc=f'Epoch {epoch+1}/{num_epochs}', leave=False) as pbar:
        for images, targets in pbar:
            images = list(image.to(device) for image in images)
            targets = [{k: v.to(device) for k, v in t.items()} for t in targets]

            loss_dict = model(images, targets)
            losses = sum(loss for loss in loss_dict.values())
            
            optimizer.zero_grad()
            losses.backward()
            optimizer.step()
            
            # 更新进度条显示的损失值
            total_loss += losses.item()
            pbar.set_postfix({'loss': f'{losses.item():.4f}'})
    
    avg_loss = total_loss / len(train_loader)
    print(f"Epoch {epoch+1}/{num_epochs} - Average loss: {avg_loss:.4f}")

# 保存模型
torch.save(model.state_dict(), "faster_rcnn.pth")
# %%
# 加载模型
model.load_state_dict(torch.load("faster_rcnn.pth"))

# 模型评估
model.eval()

# 预测val_loader的前2个数据
for images, targets in val_loader:
    images = list(image.to(device) for image in images)
    outputs = model(images)
    print(outputs)
    break
