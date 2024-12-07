# %%
import argparse
import os
from pathlib import Path
import torch
import data_setup
import model_builder
import engine
import utils
import transfer_learning
import predict

from torchvision import transforms

def parse_args():
    parser = argparse.ArgumentParser(description="训练 TinyVGG 模型进行分类任务")

    # 训练和测试目录
    parser.add_argument(
        "--train_dir",
        type=str,
        default="data/pizza_steak_sushi/train",
        help="训练数据的目录路径"
    )
    parser.add_argument(
        "--test_dir",
        type=str,
        default="data/pizza_steak_sushi/test",
        help="测试数据的目录路径"
    )

    # 超参数
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=0.001,
        help="优化器的学习率"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="训练和测试的数据批量大小"
    )
    parser.add_argument(
        "--epochs",
        type=int,
        default=5,
        help="训练的轮数"
    )
    parser.add_argument(
        "--hidden_units",
        type=int,
        default=10,
        help="TinyVGG 模型中的隐藏单元数"
    )

    args = parser.parse_args()
    return args

def main():
    args = parse_args()

    print(f"PyTorch version: {torch.__version__}")
    print(f"数据设置模块路径: {data_setup.__file__}")

    # 设置超参数
    NUM_EPOCHS = args.epochs
    BATCH_SIZE = args.batch_size
    HIDDEN_UNITS = args.hidden_units
    LEARNING_RATE = args.learning_rate

    # 数据路径
    train_dir = args.train_dir
    test_dir = args.test_dir

    # 选择设备
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    print(f"使用设备: {device}")

    
    # 构建模型
    # model = model_builder.TinyVGG(
    #     input_shape=3,
    #     hidden_units=HIDDEN_UNITS,
    #     output_shape=len(class_names)
    # ).to(device=device)
    model,auto_transforms = transfer_learning.get_model()
    print(model)
    
    # 数据转换
    data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])
    
    train_data_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.3),
        transforms.RandomRotation(30),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.2),
        transforms.RandomResizedCrop(64, scale=(0.8, 1.0)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    # 创建数据加载器
    train_dataloader, test_dataloader, class_names = data_setup.create_dataloaders(
        train_dir=train_dir,
        test_dir=test_dir,
        transform = auto_transforms,
        # test_transform=auto_transforms,
        batch_size=BATCH_SIZE
    )

    print(f"类别名称: {class_names}")


    # 定义损失函数和优化器
    loss_fn = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)

    # 开始训练
    results = engine.train(
        model=model,
        train_dataloader=train_dataloader,
        test_dataloader=test_dataloader,
        optimizer=optimizer,
        loss_fn=loss_fn,
        epochs=NUM_EPOCHS,
        device=device
    )

    # 保存模型
    utils.save_model(
        model=model,
        target_dir="models/",
        model_name="tiny_vgg.pth"
    )
    
    # Get the plot_loss_curves() function from helper_functions.py, download the file if we don't have it
    try:
        from helper_functions import plot_loss_curves
    except:
        print("[INFO] Couldn't find helper_functions.py, downloading...")
        
        with open(Path("helper_functions.py"), "wb") as f:
            import requests
            request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/helper_functions.py")
            f.write(request.content)
        from helper_functions import plot_loss_curves

    # Plot the loss curves of our model
    plot_loss_curves(results)
    
    # Get a random list of image paths from test set
    import random
    num_images_to_plot = 3
    test_image_path_list = list(Path(test_dir).glob("*/*.jpg")) # get list all image paths from test data 
    test_image_path_sample = random.sample(population=test_image_path_list, # go through all of the test image paths
                                           k=num_images_to_plot) # randomly select 'k' image paths to pred and plot

    # Make predictions on and plot the images
    for image_path in test_image_path_sample:
        predict.pred_and_plot_image(model=model, 
                            image_path=image_path,
                            class_names=class_names,
                            # transform=weights.transforms(), # optionally pass in a specified transform from our pretrained model weights
                            image_size=(224, 224))
        
    # Download custom image
    import requests

    # Setup custom image path
    # custom_image_path = Path("../data/") / "04-pizza-dad.jpeg"
    custom_image_path = Path(__file__).parent.parent / "data" / "04-pizza-dad.jpeg"

    # Download the image if it doesn't already exist
    if not custom_image_path.is_file():
        with open(custom_image_path, "wb") as f:
            # When downloading from GitHub, need to use the "raw" file link
            request = requests.get("https://raw.githubusercontent.com/mrdbourke/pytorch-deep-learning/main/images/04-pizza-dad.jpeg")
            print(f"Downloading {custom_image_path}...")
            f.write(request.content)
    else:
        print(f"{custom_image_path} already exists, skipping download.")

    # Predict on custom image
    predict.pred_and_plot_image(model=model,
                        image_path=custom_image_path,
                        class_names=class_names)

if __name__ == "__main__":
    main()
# %%