# %%
import torch
from torch import nn
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import os
from pathlib import Path
from tqdm.auto import tqdm
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.utils.class_weight import compute_class_weight
import numpy as np
from sklearn.metrics import confusion_matrix, classification_report

# 设置随机种子
torch.manual_seed(42)
if torch.backends.mps.is_available():
    torch.mps.manual_seed(42)

# 数据路径
root_dir = Path("data/") / "pizza_steak_sushi"
train_dir = root_dir / "train"
test_dir = root_dir / "test"

# 超参数
BATCH_SIZE = 16
NUM_WORKERS = 0  # 如果在Jupyter或其他交互环境中运行，设置为0避免多进程问题

# 数据转换
train_transform = transforms.Compose([
    transforms.RandomRotation(15),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(45),
    transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
    transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],  # ImageNet均值
                         std=[0.229, 0.224, 0.225])   # ImageNet标准差
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225])
])

# 加载数据集
def path_to_dataset(dir, transform):
    dataset = datasets.ImageFolder(root=dir, transform=transform)
    return dataset

train_dataset = path_to_dataset(train_dir, train_transform)
test_dataset = path_to_dataset(test_dir, test_transform)

# 计算类别权重
classes = train_dataset.classes
class_counts = [0] * len(classes)
for _, label in train_dataset.samples:
    class_counts[label] += 1

class_weights = compute_class_weight(class_weight='balanced',
                                    classes=np.unique(train_dataset.targets),
                                    y=train_dataset.targets)
class_weights = torch.tensor(class_weights, dtype=torch.float)

# 创建 WeightedRandomSampler
samples_weight = class_weights[train_dataset.targets]
sampler = WeightedRandomSampler(weights=samples_weight,
                                num_samples=len(samples_weight),
                                replacement=True)

train_dataloader = DataLoader(train_dataset,
                              batch_size=BATCH_SIZE,
                              sampler=sampler,
                              num_workers=NUM_WORKERS,
                              drop_last=True)  # 确保批量大小一致

test_dataloader = DataLoader(test_dataset,
                             batch_size=BATCH_SIZE,
                             shuffle=False,
                             num_workers=NUM_WORKERS)

# 定义模型（迁移学习示例）
class TransferLearningModel(nn.Module):
    def __init__(self, output_shape):
        super().__init__()
        # 加载预训练的ResNet18模型
        self.model = models.resnet18(pretrained=True)
        # 冻结所有参数
        for param in self.model.parameters():
            param.requires_grad = False
        # 修改最后的全连接层，移除 BatchNorm1d
        num_features = self.model.fc.in_features
        self.model.fc = nn.Sequential(
            nn.Linear(num_features, 512),
            nn.ReLU(),
            nn.Dropout(0.5),
            nn.Linear(512, output_shape)
        )
        
    def forward(self, X):
        return self.model(X)

# 定义训练步骤
def train_step(model: nn.Module,
               dataloader: DataLoader,
               loss_fn: nn.Module,
               optimizer: torch.optim.Optimizer,
               device: torch.device):
    model.train()
    train_loss, train_correct = 0, 0
    total = 0
    for batch, (X, y) in enumerate(dataloader):
        X, y = X.to(device), y.to(device)
        y_pred = model(X)
        loss = loss_fn(y_pred, y)
        train_loss += loss.item()
        preds = torch.argmax(torch.softmax(y_pred, dim=1), dim=1)
        train_correct += (preds == y).sum().item()
        total += y.size(0)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
    train_loss /= len(dataloader)
    train_acc = train_correct / total
    return train_loss, train_acc

# 定义测试步骤
def test_step(model: nn.Module,
              dataloader: DataLoader,
              loss_fn: nn.Module,
              device: torch.device):
    model.eval()
    test_loss, test_correct = 0, 0
    total = 0
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            test_pred = model(X)
            loss = loss_fn(test_pred, y)
            test_loss += loss.item()
            preds = torch.argmax(torch.softmax(test_pred, dim=1), dim=1)
            test_correct += (preds == y).sum().item()
            total += y.size(0)
    test_loss /= len(dataloader)
    test_acc = test_correct / total
    return test_loss, test_acc

# 定义EarlyStopping类
class EarlyStopping:
    def __init__(self, patience=10, verbose=False, delta=0):
        self.patience = patience
        self.verbose = verbose
        self.delta = delta
        self.counter = 0
        self.best_loss = None
        self.early_stop = False
        
    def __call__(self, val_loss):
        if self.best_loss is None:
            self.best_loss = val_loss
        elif val_loss > self.best_loss - self.delta:
            self.counter += 1
            if self.verbose:
                print(f'EarlyStopping counter: {self.counter} out of {self.patience}')
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_loss = val_loss
            self.counter = 0

# 定义训练循环
def train(train_dataloader, test_dataloader, model, loss_fn, optimizer, scheduler, device, Epochs=5):
    epochs = Epochs
    results = {"train_loss": [], "train_acc": [], "test_loss": [], "test_acc": []}
    early_stopping = EarlyStopping(patience=10, verbose=True)
    
    for epoch in tqdm(range(epochs)):
        train_loss, train_acc = train_step(model, train_dataloader, loss_fn, optimizer, device)
        test_loss, test_acc = test_step(model, test_dataloader, loss_fn, device)
        scheduler.step()
        
        print(f"Epoch:{epoch+1} | "
              f"train_loss:{train_loss:.4f} | "
              f"train_acc:{train_acc:.4f} | "
              f"test_loss:{test_loss:.4f} | "
              f"test_acc:{test_acc:.4f}")
        
        results["train_loss"].append(train_loss)
        results["train_acc"].append(train_acc)
        results["test_loss"].append(test_loss)
        results["test_acc"].append(test_acc)
        
        # 检查EarlyStopping
        early_stopping(test_loss)
        if early_stopping.early_stop:
            print("Early stopping")
            break
    
    return results

# 定义绘图函数
def plot_results(results):
    epochs = range(1, len(results['train_loss']) + 1)
    
    plt.figure(figsize=(12,5))
    
    # 绘制损失曲线
    plt.subplot(1,2,1)
    plt.plot(epochs, results['train_loss'], 'bo-', label='Training Loss')
    plt.plot(epochs, results['test_loss'], 'ro-', label='Validation Loss')
    plt.title('Training and Validation Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()
    
    # 绘制准确率曲线
    plt.subplot(1,2,2)
    plt.plot(epochs, results['train_acc'], 'bo-', label='Training Accuracy')
    plt.plot(epochs, results['test_acc'], 'ro-', label='Validation Accuracy')
    plt.title('Training and Validation Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy')
    plt.legend()
    
    plt.show()

# 定义模型评估函数，包含混淆矩阵和预测错误的样本可视化
def evaluate_model(model, dataloader, device, classes, num_errors=5):
    model.eval()
    all_preds = []
    all_labels = []
    misclassified_images = []
    misclassified_preds = []
    misclassified_labels = []
    
    with torch.inference_mode():
        for X, y in dataloader:
            X, y = X.to(device), y.to(device)
            preds = torch.argmax(torch.softmax(model(X), dim=1), dim=1)
            all_preds.extend(preds.cpu().numpy())
            all_labels.extend(y.cpu().numpy())
            
            # 收集预测错误的样本
            incorrect = preds != y
            if incorrect.any():
                misclassified_images.extend(X[incorrect].cpu())
                misclassified_preds.extend(preds[incorrect].cpu())
                misclassified_labels.extend(y[incorrect].cpu())
    
    # 混淆矩阵
    cm = confusion_matrix(all_labels, all_preds)
    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.show()
    
    # 分类报告
    cr = classification_report(all_labels, all_preds, target_names=classes)
    print("Classification Report:\n", cr)
    
    # 可视化预测错误的样本
    if len(misclassified_images) > 0:
        print(f"Displaying {min(num_errors, len(misclassified_images))} misclassified samples:")
        plt.figure(figsize=(15, 5))
        for i in range(min(num_errors, len(misclassified_images))):
            img = misclassified_images[i].permute(1, 2, 0)
            img = img * torch.tensor([0.229, 0.224, 0.225]) + torch.tensor([0.485, 0.456, 0.406])  # 反标准化
            img = torch.clamp(img, 0, 1)
            plt.subplot(1, num_errors, i+1)
            plt.imshow(img.numpy())
            plt.title(f"True: {classes[misclassified_labels[i]]}\nPred: {classes[misclassified_preds[i]]}")
            plt.axis(False)
        plt.show()
    else:
        print("No misclassified samples found.")

# 实例化模型
output_shape = 3  # 三类分类
mymodel = TransferLearningModel(output_shape=output_shape)
device = torch.device('mps') if torch.backends.mps.is_available() else torch.device('cpu')
mymodel = mymodel.to(device)
print(mymodel)

# 定义损失函数和优化器
loss_fn = nn.CrossEntropyLoss(weight=class_weights.to(device))
optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, mymodel.parameters()), lr=1e-4)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=30, gamma=0.1)

# 开始训练
EPOCHS = 100
results = train(train_dataloader, test_dataloader, mymodel, loss_fn, optimizer, scheduler, device, Epochs=EPOCHS)

# 绘制训练结果
plot_results(results)

# 评估模型
# %%
evaluate_model(mymodel, test_dataloader, device, classes, num_errors=9)
# %%
