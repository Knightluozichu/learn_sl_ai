# %% 论文复刻

# 下载论文
# Implement it
# Kepp doing this until you have skills

# 来源
# arXiv
# AK Twitter
# Papers with Code

#  我们将复制机器学习研究论文 An Image is Worth 16x16 Words： Transformers for Image Recognition at Scale（使用 PyTorch 进行大规模图像识别的转换器）
# Transformer 神经网络架构最初是在机器学习研究论文 Attention is all you need 中引入的
# 顾名思义，Vision Transformer （ViT） 架构旨在使原始的 Transformer 架构适应视觉问题（分类是第一个，因为许多其他分类都效仿）。
# 最初的 Vision Transformer 在过去几年中经历了几次迭代，但是，我们将专注于复制原版，也称为“vanilla Vision Transformer”。因为如果你能重现原件，你就可以适应其他的。
# %%
# 开始设置
# 获取数据
# 创建数据集和DataLoader
# 复制ViT论文：概述
# The Path  Embedding
# 多头注意力
# 多层感知机
# 创建Transformer编码器
# 将他们放在一起创建ViT
# 为我们的ViT模型设置训练代码
# 使用来自torchvision.models的预训练ViT
# 对自定义图像进行预测
# %%
# 检查模块


try:
    import torch
    import torchvision
    assert int(torch.__version__.split('.')[1]) >= 12 or int(torch.__version__.split('.')[0]) == 2, "torch version should be 1.12+"
    assert int(torchvision.__version__.split('.')[1]) >= 13, "torchvision version should be 0.13+"
    print(f"torch version:{torch.__version__}")
    print(f"torchvision version:{torchvision.__version__}")
except:
    print(f"[INFO] torch/torchvision versions not as required, installing nightly versions.")
    !pip3 install -U torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
    import torch
    import torchvision
    print(f"torch version:{torch.__version__}")
    print(f"torchvision version:{torchvision.__version__}")   
# %%
# Continue with regular imports
import matplotlib.pyplot as plt
import torch
import torchvision

from torch import nn
from torchvision import transforms

# Try to get torchinfo, install it if it doesn't work
try:
    from torchinfo import summary
except:
    print("[INFO] Couldn't find torchinfo... installing it.")
    !pip install -q torchinfo
    from torchinfo import summary

# Try to import the going_modular directory, download it from GitHub if it doesn't work
try:
    from going_modular import data_setup, engine
    from helper_functions import download_data, set_seeds, plot_loss_curves
except:
    # Get the going_modular scripts
    print("[INFO] Couldn't find going_modular or helper_functions scripts... downloading them from GitHub.")
    !git clone https://github.com/mrdbourke/pytorch-deep-learning
    !mv pytorch-deep-learning/going_modular .
    !mv pytorch-deep-learning/helper_functions.py . # get the helper_functions.py script
    !rm -rf pytorch-deep-learning
    from going_modular import data_setup, engine
    from helper_functions import download_data, set_seeds, plot_loss_curves
# %%
# 编写设备无关代码
device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
device

# %%
# 获取数据
image_path = download_data(source="https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")
image_path
# %%
# 创建 datasets and dataloaders
from going_modular.data_setup import create_dataloaders

train_dir = image_path / "train"
test_dir = image_path / "test"

IMAGE_SIZE=224

manual_transforms = transforms.Compose(
    [transforms.Resize((IMAGE_SIZE,IMAGE_SIZE)),
    transforms.ToTensor()]
)

BATCH_SIZE=32

train_dataloader,test_dataloader,class_names = create_dataloaders(train_dir=train_dir,
                                                                  test_dir=test_dir,
                                                                  transform=manual_transforms,
                                                                  batch_size=BATCH_SIZE)

train_dataloader, test_dataloader, class_names
# %%
# visualize,visualize,visualize
image_batch,label_batch = next(iter(train_dataloader))

image,label = image_batch[0], label_batch[0]

image.shape, label
# %%
# visualize,visualize,visualize
plt.imshow(image.permute(1,2,0))
plt.title(class_names[label])
plt.axis(False)
# %%
# replicating the ViT paper
# model input : images of pizza, steak and sushi
# model output:predicted labels of pizza, steak or sushi
# 具体化：ViT 是由什么制成的？
# 我们将为架构设计寻找的三个主要资源是:
# 这从图形意义上给出了模型的概述，您几乎可以单独使用此图重新创建架构
# 第 3.1 节中的四个方程 - 这些方程为图 1 中的彩色块提供了更多的数学基础
# 此表显示了不同 ViT 模型变体的各种超参数设置（例如层数和隐藏单元数）。我们将专注于最小的版本 ViT-Base。
# ViT 架构由几个阶段组成:
# 补丁 + 位置嵌入（输入）- 将输入图像转换为一系列图像补丁，并添加位置编号以指定补丁的进入顺序
# 扁平化补丁的线性投影 （Embedded Patches） - 图像补丁被转换为嵌入，使用嵌入而不仅仅是图像值的好处是，嵌入是图像的可学习表示（通常以向量的形式），可以通过训练进行改进。
# Norm - 这是“Layer Normalization”或“LayerNorm”的缩写，是一种正则化（减少过度拟合）神经网络的技术，您可以通过 PyTorch 层 torch.nn.LayerNorm() 使用 LayerNorm。
# 多头注意力 - 这是一个多头自注意力层，简称“MSA”。您可以通过 PyTorch 层 torch.nn.MultiheadAttention() 创建 MSA 层。
# MLP（或多层感知器）- MLP 通常可以引用任何前馈层集合（或者在 PyTorch 的情况下，可以引用具有 forward() 方法的层集合）。在 ViT 论文中，作者将 MLP 称为“MLP 块”，它包含两个 torch.nn.Linear() 层，它们之间有一个 torch.nn.GELU() 非线性激活（第 3.1 节），每个层之后都有一个 torch.nn.Dropout() 层（附录 B.1）
# Transformer Encoder - Transformer 编码器是上面列出的层的集合。Transformer 编码器内部有两个 skip 连接（“+”符号），这意味着该层的输入直接馈送到直接层以及后续层。整个 ViT 架构由许多相互堆叠的 Transformer 编码器组成。
# MLP 头 - 这是架构的输出层，它将学习到的输入特征转换为类输出。由于我们正在进行图像分类，因此您也可以将其称为 “classifier head”。MLP 头的结构类似于 MLP 块。

height = 224
width = 224
color_channels = 3
patch_size = 16

number_of_patches = int((height*width) / patch_size ** 2)
print(f"Number of patches (N) with image height (H={height}), width (W={width}) and patch size (P={patch_size}): {number_of_patches}")
# %%
embedding_layer_input_shape=(height,width,color_channels)
embedding_layer_output_shape = (number_of_patches, patch_size**2 * color_channels)
print(f"Input shape (single 2D image): {embedding_layer_input_shape}")
print(f"Output shape (single 2D image flattened into patches): {embedding_layer_output_shape}")

# %%
# 将单个镜像转换为补丁
plt.imshow(image.permute(1,2,0))
plt.title(class_names[label])
plt.axis(False)

# %%
image_permuted = image.permute(1,2,0)

patch_size = 16
plt.figure(figsize=(patch_size, patch_size))
plt.imshow(image_permuted[:patch_size,:,:])
# %%
img_size = 224
patch_size = 16
num_patches = img_size / patch_size
assert img_size % patch_size ==0, "Image size must be divisible by patch size."
print(f"Number of patches per row:{num_patches}\nPatch size:{patch_size} pixels x {patch_size} pixels")

fig, axs = plt.subplots(nrows=1,
                        ncols=img_size // patch_size,
                        figsize=(num_patches, num_patches),
                        sharex=True,
                        sharey=True)

for i,patch in enumerate(range(0, img_size, patch_size)):
    axs[i].imshow(image_permuted[:patch_size, patch:patch+patch_size,:])
    axs[i].set_xlabel(i+1) # set the label
    axs[i].set_xticks([])
    axs[i].set_yticks([])
# %%
img_size = 224
patch_size = 16
num_patches = img_size / patch_size
assert img_size % patch_size == 0, "Image size must be divisible by patch size."
print(f"Number of patches per row: {num_patches}\
    \nNumber of patches per column: {num_patches}\
    \nTotal patches: {num_patches*num_patches}\
    \nPatch size: {patch_size} pixels x {patch_size} pixels.")

fig, axs = plt.subplots(nrows=img_size // patch_size,
                        ncols=img_size // patch_size,
                        figsize=(num_patches, num_patches),
                        sharex=True,
                        sharey=True)

for i, patch_height in enumerate(range(0, img_size, patch_size)): # iterate through height
    for j, patch_width in enumerate(range(0, img_size, patch_size)): # iterate through width

        # Plot the permuted image patch (image_permuted -> (Height, Width, Color Channels))
        axs[i, j].imshow(image_permuted[patch_height:patch_height+patch_size, # iterate through height
                                        patch_width:patch_width+patch_size, # iterate through width
                                        :]) # get all color channels

        # Set up label information, remove the ticks for clarity and set labels to outside
        axs[i, j].set_ylabel(i+1,
                             rotation="horizontal",
                             horizontalalignment="right",
                             verticalalignment="center")
        axs[i, j].set_xlabel(j+1)
        axs[i, j].set_xticks([])
        axs[i, j].set_yticks([])
        axs[i, j].label_outer()

# Set a super title
fig.suptitle(f"{class_names[label]} -> Patchified", fontsize=16)
plt.show()
# %% 现在我们如何将这些 patch 中的每一个转换为 embedding 并将它们转换为 sequence 呢？
from torch import nn

patch_size =16

conv2d = nn.Conv2d(in_channels=3,
                   out_channels=768,
                   kernel_size=patch_size,
                   stride=patch_size,
                   padding=0)

plt.imshow(image.permute(1,2,0))
plt.title(class_names[label])
plt.axis(False)

# %%
image_out_of_conv = conv2d(image.unsqueeze(0))
print(image_out_of_conv.shape)
# %%
import random
random_indexes = random.sample(range(0, 758), k=5)
print(f"Showing random convolutional feature maps from indexes:{random_indexes}")

fig, axs = plt.subplots(nrows=1,
                        ncols=5,
                        figsize=(12,12))
for i,idx in enumerate(random_indexes):
    image_conv_feature_map = image_out_of_conv[:, idx, :, :]
    axs[i].imshow(image_conv_feature_map.squeeze().detach().numpy())
    axs[i].set(xticklabels=[], yticklabels=[], xticks=[], yticks=[])

# %%
single_feature_map = image_out_of_conv[:,0,:,:]
single_feature_map, single_feature_map.requires_grad

# %%
# Current tensor shape
print(f"Current tensor shape: {image_out_of_conv.shape} -> [batch, embedding_dim, feature_map_height, feature_map_width]")
# %%
flatten = nn.Flatten(start_dim=2,
                     end_dim=3)

# %%
# 1. View single image
plt.imshow(image.permute(1, 2, 0)) # adjust for matplotlib
plt.title(class_names[label])
plt.axis(False);
print(f"Original image shape: {image.shape}")

# 2. Turn image into feature maps
image_out_of_conv = conv2d(image.unsqueeze(0)) # add batch dimension to avoid shape errors
print(f"Image feature map shape: {image_out_of_conv.shape}")

# 3. Flatten the feature maps
image_out_of_conv_flattened = flatten(image_out_of_conv)
print(f"Flattened image feature map shape: {image_out_of_conv_flattened.shape}")
# %%
image_out_of_conv_flattened_reshaped = image_out_of_conv_flattened.permute(0,2,1)
print(f"Patch embedding sequence shape: {image_out_of_conv_flattened_reshaped.shape}")
# %%
# Get a single flattened feature map
single_flattened_feature_map = image_out_of_conv_flattened_reshaped[:, :, 0] # index: (batch_size, number_of_patches, embedding_dimension)

# Plot the flattened feature map visually
plt.figure(figsize=(22, 22))
plt.imshow(single_flattened_feature_map.detach().numpy())
plt.title(f"Flattened feature map shape: {single_flattened_feature_map.shape}")
plt.axis(False)
# %%
class PatchEmbedding(nn.Module):
    def __init__(self,
                 in_channels:int=3,
                 patch_size:int=16,
                 embedding_dim:int=768):
        super().__init__()
        
        self.patcher = nn.Conv2d(in_channels=in_channels,
                                 out_channels=embedding_dim,
                                 kernel_size=patch_size,
                                 stride=patch_size,
                                 padding=0)
        self.flatten = nn.Flatten(start_dim=2,
                                  end_dim=3)
        
    def forward(self, x):
        image_resolution=x.shape[-1]
        assert image_resolution % patch_size == 0, f"Input image size must be divisible by patch size, image shape: {image_resolution}, patch size: {patch_size}"

        x_patched = self.patcher(x)
        x_flattened = self.flatten(x_patched)
        
        return x_flattened.permute(0,2,1)

set_seeds()

patchify = PatchEmbedding(in_channels=3,
                          patch_size=16,
                          embedding_dim=768)

# Pass a single image through
print(f"Input image shape: {image.unsqueeze(0).shape}")
patch_embedded_image = patchify(image.unsqueeze(0)) # add an extra batch dimension on the 0th index, otherwise will error
print(f"Output patch embedding shape: {patch_embedded_image.shape}")

# %%
# Create random input sizes
random_input_image = (1, 3, 224, 224)
random_input_image_error = (1, 3, 250, 250) # will error because image size is incompatible with patch_size

# # Get a summary of the input and outputs of PatchEmbedding (uncomment for full output)
summary(PatchEmbedding(),
        input_size=random_input_image, # try swapping this for "random_input_image_error"
        col_names=["input_size", "output_size", "num_params", "trainable"],
        col_width=20,
        row_settings=["var_names"])
# %%
# 好了，我们已经完成了图像补丁嵌入，是时候开始处理类标记嵌入了。

print(patch_embedded_image)
print(f"Patch embedding shape: {patch_embedded_image.shape} -> [batch_size, number_of_patches, embedding_dimension] ")

# %%
batch_size = patch_embedded_image.shape[0]
embedding_dimension = patch_embedded_image.shape[-1]

class_token = nn.Parameter(torch.ones(batch_size,
                                      1,
                                      embedding_dimension),
                           requires_grad=True)
print(class_token[:,:,:10])

print((f"Class token shape:{class_token.shape} -> [batch_size, number_of_tokens, embedding_dimension]"))

# %%
patch_embedded_image_with_class_embedding = torch.cat((class_token,
                                                       patch_embedded_image),
                                                      dim=1)
print(patch_embedded_image_with_class_embedding)
print((f"Sequence of patch embeddings with class token prepended shape: {patch_embedded_image_with_class_embedding.shape} ->[batch_size, number_of_patches,embedding_dimension]"))

# %%
patch_embedded_image_with_class_embedding, patch_embedded_image_with_class_embedding.shape
# %%
# 计算N（number of patches)
number_of_patches = int((height * width) / patch_size ** 2)

# 得到embedding维度
embedding_dimension = patch_embedded_image_with_class_embedding.shape[2]

# 创建可学习的1D的位置embedding
position_embedding = nn.Parameter(torch.ones(1,
                                             number_of_patches+1,
                                             embedding_dimension))

print(position_embedding[:,:10,:10])
print(f"Position embedding shape: {position_embedding.shape}")


# %%
patch_and_position_embedding = patch_embedded_image_with_class_embedding + position_embedding
print(patch_and_position_embedding)
print(f"Patch embeddings, class token prepended and positional embeddings added shape: {patch_and_position_embedding.shape} -> [batch_size, number_of_patches, embedding_dimension]")

# %%
set_seeds()
patch_size = 16
print(f"Image tensor shape: {image.shape}")
height, width = image.shape[1], image.shape[2]

x = image.unsqueeze(0)
print(f"Input image with batch dimension shape: {x.shape}")

patch_embedding_layer = PatchEmbedding(in_channels=3,
                                       patch_size = patch_size,
                                       embedding_dim=768)
patch_embedding = patch_embedding_layer(x)
print(f"Patching embedding shape: {patch_embedding.shape}")

batch_size = patch_embedding.shape[0]
embedding_dimension = patch_embedding.shape[-1]

patch_embedding_class_token = torch.cat((class_token,
                                         patch_embedding),
                                        dim=1)
print(f"Patch embedding with class token shape: {patch_embedding_class_token.shape}")

number_of_patches = int((height*width) / patch_size**2)
position_embedding = nn.Parameter(torch.ones(1,number_of_patches+1,embedding_dimension),requires_grad=True)
patch_and_position_embedding = patch_embedding_class_token + position_embedding
print(f"Patch and position embedding shape: {patch_and_position_embedding.shape}")
# %% 等式 2：多头注意力 （MSA)

class MultiheadSelfAttentionBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 attn_dropout:float=0):
        super().__init__()
        self.layer_norm=nn.LayerNorm(normalized_shape=embedding_dim)
        self.multihead_attn = nn.MultiheadAttention(embed_dim=embedding_dim,
                                                    num_heads=num_heads,
                                                    dropout=attn_dropout,
                                                    batch_first=True)
    def forward(self, x):
        x = self.layer_norm(x)
        atten_output, _ = self.multihead_attn(query=x,
                                              key=x,
                                              value=x,
                                              need_weights = False)
        return atten_output


# %%
multihead_self_attention_block = MultiheadSelfAttentionBlock(embedding_dim=768,
                                                             num_heads=12)
patched_image_through_msa_block=multihead_self_attention_block(patch_and_position_embedding)

print(f"Input shape of MSA blcok:{patch_and_position_embedding.shape}")
print(f"Output shape MSA blcok: {patched_image_through_msa_block.shape}")

# %% 等式 3：多层感知器 （MLP）
# layer norm -> linear layer -> non-linear layer -> dropout -> linear layer -> dropout

class MLPBlock(nn.Module):
    
    def __init__(self,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 dropout:float=0.1
                 ):
        super().__init__()
        self.layer_norm = nn.LayerNorm(normalized_shape=embedding_dim)
        
        self.mlp = nn.Sequential(
            nn.Linear(in_features=embedding_dim,
                      out_features=mlp_size),
            nn.GELU(),
            nn.Dropout(p=dropout),
            nn.Linear(in_features=mlp_size,
                      out_features=embedding_dim),
            nn.Dropout(p=dropout)
        )
        
    def forward(self, x):
        x = self.layer_norm(x)
        x = self.mlp(x)
        return x


# %%
mlp_block = MLPBlock(embedding_dim=768,
                     mlp_size=3072,
                     dropout=0.1)
patched_image_through_mlp_block = mlp_block(patched_image_through_msa_block)
print(f"Input shape of MLP block: {patched_image_through_msa_block.shape}")
print(f"Output shape MLP block: {patched_image_through_mlp_block.shape}")
# %% x_input -> MSA_block -> [MSA_block_output + x_input] -> MLP_block -> [MLP_block_output + MSA_block_output + x_input] -> ...

class TransformerEncoderBlock(nn.Module):
    def __init__(self,
                 embedding_dim:int=768,
                 num_heads:int=12,
                 mlp_size:int=3072,
                 mlp_dropout:float=0.1,
                 attn_dropout:float=0):
        super().__init__()

        self.msa_block = MultiheadSelfAttentionBlock(embedding_dim=embedding_dim,
                                                     num_heads=num_heads,
                                                     attn_dropout=attn_dropout)
        
        self.mlp_block = MLPBlock(embedding_dim=embedding_dim,
                                  mlp_size=mlp_size,
                                  dropout=mlp_dropout)
        
    def forward(self, x):
        x = self.msa_block(x) + x
        x = self.mlp_block(x) + x
        return x

# %%
transforms_encoder_block = TransformerEncoderBlock()

# summary(model=transforms_encoder_block,
#         input_size=(1,197,768),
#         col_names=["input_size","output_size","num_params","trainable"],
#         col_width=20,
#         row_settings=["var_names"])

# %%
torch_transformer_encoder_layer = nn.TransformerEncoderLayer(d_model=768,
                                                             nhead=12,
                                                             dim_feedforward=3072,
                                                             dropout=0.1,
                                                             activation="gelu",
                                                             batch_first=True,
                                                             norm_first=True)
torch_transformer_encoder_layer
# %%

summary(model=torch_transformer_encoder_layer,
        input_size=(1,197,768),
        col_names=["input_size","output_size","num_params","trainable"],
        col_width=20,
        row_settings=["var_names"])
# %
# %%
class ViT(nn.Module):
    def __init__(self,
                 img_size:int=224,
                 in_channels:int=3,
                 patch_size:int=16,
                 num_transformer_layer:int=12,
                 embedding_dim:int=768,
                 mlp_size:int=3072,
                 num_heads:int=12,
                 attn_dropout:float=0,
                 mlp_dropout:float=0.1,
                 embedding_dropout:float=0.1,
                 num_classes:int=1000):
        super().__init__()
        
        assert img_size % patch_size ==0, f"Image size must be divisible by patch size, image size: {img_size}, patch size:{patch_size}"
        
        self.num_patches = (img_size * img_size) // patch_size**2
        self.class_embedding = nn.Parameter(data=torch.randn(1,1,embedding_dim),
                                            requires_grad=True)
        self.position_embedding = nn.Parameter(data=torch.randn(1,self.num_patches + 1, embedding_dim),
                                               requires_grad=True)
        self.embedding_dropout = nn.Dropout(p=embedding_dropout)
        self.patch_embedding = PatchEmbedding(in_channels=in_channels,
                                              patch_size=patch_size,
                                              embedding_dim=embedding_dim)
        self.transformer_encoder = nn.Sequential(*[TransformerEncoderBlock(embedding_dim=embedding_dim,
                                                                           num_heads=num_heads,
                                                                           mlp_size=mlp_size,
                                                                           mlp_dropout=mlp_dropout,
                                                                           attn_dropout=attn_dropout) for _ in range(num_transformer_layer)])
        self.classifier = nn.Sequential(
            nn.LayerNorm(normalized_shape=embedding_dim),
            nn.Linear(in_features=embedding_dim,
                      out_features=num_classes)
        )
    
    def forward(self, x):
        batch_size  = x.shape[0]
        class_token = self.class_embedding.expand(batch_size,
                                                  -1,-1)
        x = self.patch_embedding(x)
        x = torch.cat((class_token,x),dim=1)
        x=self.position_embedding +x
        x = self.embedding_dropout(x)
        x=self.transformer_encoder(x)
        x =self.classifier(x[:,0])
        return x

# %%
batch_size =32
class_token_embedding_single = nn.Parameter(data=torch.randn(1,1,768))
class_token_embedding_expanded = class_token_embedding_single.expand(batch_size, -1, -1) # expand the single learnable class token across the batch dimension, "-1" means to "infer the dimension"

# Print out the change in shapes
print(f"Shape of class token embedding single: {class_token_embedding_single.shape}")
print(f"Shape of class token embedding expanded: {class_token_embedding_expanded.shape}")

# %%
set_seeds()

# Create a random tensor with same shape as a single image
random_image_tensor = torch.randn(1, 3, 224, 224) # (batch_size, color_channels, height, width)

# Create an instance of ViT with the number of classes we're working with (pizza, steak, sushi)
# len(class_names)
vit = ViT(num_classes= len(class_names))

# vit = ViT(num_classes= 1000)

# Pass the random image tensor to our ViT instance
vit(random_image_tensor)
# %%
from torchinfo import summary

summary(model=vit,
        input_size=(32,3,224,224),
        col_names=["input_size","output_size","num_params","trainable"],
        col_width=20,
        row_settings=["var_names"])
# %%
from going_modular import engine

optimizer = torch.optim.Adam(params=vit.parameters(),
                             lr=3e-3,
                             betas=(0.9, 0.999),
                             weight_decay=0.3)
loss_fn = torch.nn.CrossEntropyLoss()
set_seeds()
results=engine.train(model=vit,
                     train_dataloader=train_dataloader,
                     test_dataloader=test_dataloader,
                     optimizer=optimizer,
                     loss_fn=loss_fn,
                     epochs=10,
                     device=device)

# %%
from helper_functions import plot_loss_curves

plot_loss_curves(results)

# %% 与论文比较效果差的原因
# Hyperparameter value 超参数值	ViT Paper ViT 纸	Our implementation 我们的实施
# Number of training images 训练图像的数量	1.3M (ImageNet-1k), 14M (ImageNet-21k), 303M (JFT)
# 1.3M （ImageNet-1k）、14M （ImageNet-21k）、303M （JFT）
# 225
# Epochs 时代	7 (for largest dataset), 90, 300 (for ImageNet)
# 7 （对于最大的数据集）， 90， 300 （对于 ImageNet）
# 10
# Batch size 批量大小	4096	32
# Learning rate warmup 学习率预热	10k steps (Table 3) 10k 步（表 3）	None 没有
# Learning rate decay 学习率衰减	Linear/Cosine (Table 3) 线性/余弦（表 3）	None 没有
# Gradient clipping 渐变剪裁	Global norm 1 (Table 3) 全球规范 1（表 3）	None 没有

# 怎么办？
# 迁移学习，用huggingface，timm，paperwithcode上的预训练模型

import torch
import torchvision
print((torch.__version__))
print((torchvision.__version__))

# %%
device = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
device
# %%
pretrained_vit_weights = torchvision.models.ViT_B_16_Weights.DEFAULT
pretrained_vit = torchvision.models.vit_b_16(weights=pretrained_vit_weights)

for parameter in pretrained_vit.parameters():
    parameter.requires_grad=False

set_seeds()
pretrained_vit.heads= nn.Linear(in_features=768, out_features=len(class_names)).to(device=device)

# %%
# download data
from helper_functions import download_data

image_path = download_data(source="https://github.com/mrdboutke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip",
                           destination="pizza_steak_sushi")
image_path
# %%
train_dir = image_path / "train"
test_dir = image_path / "test"
train_dir, test_dir
# %%
# transforms要一样
pretrained_vit_transforms = pretrained_vit_weights.transforms()
print(pretrained_vit_transforms)
# %%
from going_modular import data_setup
from going_modular import engine

train_dataloader_pretrained, test_dataloader_pretrained, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                                     test_dir=test_dir,
                                                                                                     transform=pretrained_vit_transforms,
                                                                                                     batch_size=32)
optimizer = torch.optim.Adam(params=pretrained_vit.parameters(),
                             lr=1e-3)

set_seeds()

pretrained_vit_results = engine.train(model=pretrained_vit,
                                      train_dataloader=train_dataloader_pretrained,
                                      test_dataloader=test_dataloader_pretrained,
                                      optimizer=optimizer,
                                      loss_fn=loss_fn,
                                      epochs=10,
                                      device=device)

# %%
# Plot the loss curves
from helper_functions import plot_loss_curves

plot_loss_curves(pretrained_vit_results)
# %%
# Save the model
from going_modular import utils

utils.save_model(model=pretrained_vit,
                 target_dir="models",
                 model_name="08_pretrained_vit_feature_extractor_pizza_steak_sushi.pth")
# %%
from pathlib import Path

# Get the model size in bytes then convert to megabytes
pretrained_vit_model_size = Path("models/08_pretrained_vit_feature_extractor_pizza_steak_sushi.pth").stat().st_size // (1024*1024) # division converts bytes to megabytes (roughly)
print(f"Pretrained ViT feature extractor model size: {pretrained_vit_model_size} MB")
# %%
import requests
from going_modular import predict

custom_image_path = image_path / "04-pizza-dad.jpeg"

if not custom_image_path.is_file():
    with open(custom_image_path,"wb") as f:
        request =requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
        print(f"Downloading {custom_image_path}...")
        f.write(request.content)
else:
    print(f"{custom_image_path} already exists, skipping download.")

predict.pred_and_plot_image(model=pretrained_vit,
                    image_path=custom_image_path,
                    class_names=class_names)
# %%

# %%
