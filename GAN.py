# %%
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt

# %% 步骤2：准备数据集
# .	定义数据转换：
# 使用transforms对MNIST数据进行预处理，将图像转换为Tensor，并将像素值归一化到[-1, 1]范围：
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])
# device = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
# torch.set_default_device(device)
# .	下载MNIST数据集：
# 使用datasets.MNIST下载数据集，并将数据转换为Tensor：
mnist = datasets.MNIST(root='data', download=False, train=True, transform=transform)
dataloader = DataLoader(mnist, batch_size=64, shuffle=True)
# %%
# 步骤3：定义模型架构
# 	1.	生成器（Generator）：
# 生成器将随机噪声转换为图像。定义一个简单的全连接神经网络：

class Generator(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(Generator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 512),
            nn.LeakyReLU(0.2),
            nn.Linear(512, 1024),
            nn.LeakyReLU(0.2),
            nn.Linear(1024, output_dim),
            nn.Tanh()
        )
        
    def forward(self, x):
        x = self.model(x)
        return x
# %%
# 2.	判别器（Discriminator）：
# 判别器将图像输入并输出其为真实图像的概率。定义一个简单的全连接神经网络：

class Discriminator(nn.Module):
    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(input_dim, 1024),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(1024, 512),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(512, 256),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(256, 1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        x = self.model(x)
        return x
    
# %%
# 步骤4：定义超参数
# 定义模型超参数，包括输入维度、生成器和判别器的输入和输出维度、学习率等：
z_dim = 100
img_dim = 28 * 28
lr = 0.0002

device = "mps" if torch.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
# device = "cpu"

# 数据to device


G = Generator(input_dim=z_dim, output_dim=img_dim).to(device)
D = Discriminator(input_dim=img_dim).to(device)

criterion = nn.BCELoss()
optimizer_G = optim.Adam(G.parameters(), lr=lr)
optimizer_D = optim.Adam(D.parameters(), lr=lr)
# %%
# 步骤5：训练模型
# 	1.	训练循环：

num_epochs = 50
for epoch in range(num_epochs):
    for real_imgs, _ in dataloader:
        batch_size = real_imgs.size(0)
        real_imgs = real_imgs.view(batch_size, -1).to(device)

        # 训练判别器
        z = torch.randn(batch_size, z_dim, device=device)
        fake_imgs = G(z)
        real_labels = torch.ones(batch_size, 1).to(device)
        fake_labels = torch.zeros(batch_size, 1).to(device)

        outputs = D(real_imgs)
        d_loss_real = criterion(outputs, real_labels)
        real_score = outputs

        outputs = D(fake_imgs.detach())
        d_loss_fake = criterion(outputs, fake_labels)
        fake_score = outputs

        d_loss = d_loss_real + d_loss_fake
        optimizer_D.zero_grad()
        d_loss.backward()
        optimizer_D.step()

        # 训练生成器
        outputs = D(fake_imgs)
        g_loss = criterion(outputs, real_labels)

        optimizer_G.zero_grad()
        g_loss.backward()
        optimizer_G.step()

    print(f'Epoch [{epoch+1}/{num_epochs}], d_loss: {d_loss.item():.4f}, g_loss: {g_loss.item():.4f}, '
          f'D(x): {real_score.mean().item():.2f}, D(G(z)): {fake_score.mean().item():.2f}')
# %%
import torchvision
# 步骤6：生成和可视化图像
# 	1.	生成新图像：
z = torch.randn(batch_size, z_dim).to(device)
fake_imgs = G(z)
fake_imgs = fake_imgs.view(fake_imgs.size(0), 1, 28, 28)

# 2.	可视化生成的图像：
grid = torchvision.utils.make_grid(fake_imgs, nrow=8, normalize=True)
plt.imshow(grid.permute(1, 2, 0).cpu().detach().numpy())
plt.show()
# %% DCGAN
#%matplotlib inline
import argparse
import os
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.transforms as transforms
import torchvision.utils as vutils
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from IPython.display import HTML

# Set random seed for reproducibility
manualSeed = 999
#manualSeed = random.randint(1, 10000) # use if you want new results
print("Random Seed: ", manualSeed)
random.seed(manualSeed)
torch.manual_seed(manualSeed)
torch.use_deterministic_algorithms(True) # Needed for reproducible results
# %%
# Root directory for dataset
dataroot = "data/MNIST"

# Number of workers for dataloader
workers = 0

# Batch size during training
batch_size = 32

# Spatial size of training images. All images will be resized to this
#   size using a transformer.
image_size = 28

# Number of channels in the training images. For color images this is 3
nc = 3

# Size of z latent vector (i.e. size of generator input)
nz = 100

# Size of feature maps in generator
ngf = 28

# Size of feature maps in discriminator
ndf = 28

# Number of training epochs
num_epochs = 5

# Learning rate for optimizers
lr = 0.0002

# Beta1 hyperparameter for Adam optimizers
beta1 = 0.5

# Number of GPUs available. Use 0 for CPU mode.
ngpu = 1

# We can use an image folder dataset the way we have it setup.
# Create the dataset
dataset = dset.MNIST(root='data', download=False, train=True,transform=transforms.Compose([
                                   transforms.Resize(image_size),
                                   transforms.CenterCrop(image_size),
                                   transforms.ToTensor(),
                                   transforms.Lambda(lambda x: x.repeat(3, 1, 1)),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))

# 对于MNIST数据集，我们将使用3个通道，因为我们将使用DCGAN网络结构，它需要3个通道的输入图像。
# 所以对dataset数据集进行转换，将单通道图像转换为3通道图像。
# 添加转换以将单通道图像转换为三通道图像




# %%

# Create the dataloader
dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size,
                                         shuffle=True, num_workers=workers)

# Decide which device we want to run on
device = torch.device("cuda:0" if (torch.cuda.is_available() and ngpu > 0) else "cpu")

# Plot some training images
real_batch = next(iter(dataloader))
plt.figure(figsize=(8,8))
plt.axis("off")
plt.title("Training Images")
plt.imshow(np.transpose(vutils.make_grid(real_batch[0].to(device)[:64], padding=2, normalize=True).cpu(),(1,2,0)))
plt.show()
# %%
