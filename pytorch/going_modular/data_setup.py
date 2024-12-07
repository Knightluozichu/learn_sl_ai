import os
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

NUM_WORKERS = 0

def create_dataloaders(
    train_dir: str,
    test_dir: str,
    transform: transforms.Compose,
    # test_transform: transforms.Compose,
    batch_size: int,
    num_workers: int = NUM_WORKERS
):
    """
    从图像目录创建训练和测试数据加载器。

    参数:
        train_dir (str): 训练数据目录的路径
        test_dir (str): 测试数据目录的路径
        transform (transforms.Compose): 应用于图像的转换
        batch_size (int): 每批加载的样本数量
        num_workers (int, optional): 用于数据加载的子进程数。默认为 NUM_WORKERS

    返回:
        tuple: 包含以下内容的元组:
            - train_dataloader (DataLoader): 训练数据的数据加载器
            - test_dataloader (DataLoader): 测试数据的数据加载器
            - class_names (list): 类别名称列表
    """
    train_data = datasets.ImageFolder(train_dir, transform=transform)
    # print(len(train_data))
    test_data = datasets.ImageFolder(test_dir,transform=transform)
    class_names = train_data.classes
    train_dataloader = DataLoader(
        train_data,
        batch_size = batch_size,
        shuffle=True,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    test_dataloader = DataLoader(
        test_data,
        batch_size=batch_size,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=True
    )
    return train_dataloader,test_dataloader,class_names

