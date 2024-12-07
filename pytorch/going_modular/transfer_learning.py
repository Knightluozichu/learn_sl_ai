

def get_model():
    
    # What is transfer learning?¶ 什么是迁移学习？¶
    # Transfer learning allows us to take the patterns (also called weights) another model has learned from another problem and use them for our own problem.
    # 迁移学习允许我们获取另一个模型从另一个问题中学到的模式（也称为权重），并将它们用于我们自己的问题。

    # 例如，我们可以采用计算机视觉模型从 ImageNet 等数据集（数百万张不同对象的图像）中学习的模式，
    # 并使用它们来支持我们的 FoodVision Mini 模型。

    import torchvision


    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT

    print(weights)

    # Get the transforms used to create our pretrained weights
    auto_transforms = weights.transforms()
    print(auto_transforms)

    import torch

    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

    model = torchvision.models.efficientnet_b0(weights=weights).to(device)

    import torchinfo

    torchinfo.summary(model, input_size=(32, 3, 224, 224),
                        col_names=["input_size", "output_size", "num_params", "trainable"],
                        col_width=20,
                        row_settings=["var_names"])

    # Set the manual seeds
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)

    for param in model.features.parameters():
        param.requires_grad = False

    # Get the length of class_names (one output unit for each class)
    output_shape = 3

    # Recreate the classifier layer and seed it to the target device
    model.classifier = torch.nn.Sequential(
        torch.nn.Dropout(p=0.2, inplace=True), 
        torch.nn.Linear(in_features=1280, 
                        out_features=output_shape, # same number of output units as our number of classes
                        bias=True)).to(device)

    torchinfo.summary(model, input_size=(32, 3, 224, 224),
                        col_names=["input_size", "output_size", "num_params", "trainable"],
                        col_width=20,
                        row_settings=["var_names"])
    
    model = model.to(device)
    return model,auto_transforms 