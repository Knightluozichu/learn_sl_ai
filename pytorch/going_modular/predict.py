import pathlib 
import torch
import os
from PIL import Image
from torchvision import transforms

# if "__file__" in globals():
#     import os, sys
#     sys.path.append(os.path.join(os.path.dirname(__file__),".."))

import going_modular.transfer_learning
import going_modular.model_builder

def load_model(model_path:str,
                model:torch.nn.Module):
     """
     加载模型的权重。
     
     参数:
          model_path (str): 模型权重的路径
          model (torch.nn.Module): 要加载权重的模型
     """
     if os.path.exists(model_path):
          model.load_state_dict(torch.load(model_path, weights_only=False))
          print(f"[INFO] Model loaded from {model_path}")
     else:
          print(f"[ERROR] Model path {model_path} does not exist")

def predict_image(image_path:str,
                  model:torch.nn.Module,
                  class_names:list,
                  data_transforms:transforms.Compose=None):
    """
    预测图像的类别。
    
    参数:
        image_path (str): 图像文件的路径
        model (torch.nn.Module): 训练好的模型
        class_names (list): 类别名称列表
    
    返回:
        str: 预测的类别名称
    """
    if data_transforms is None:
        data_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])
    
    device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"
    
    image = Image.open(image_path)
    image = data_transform(image).unsqueeze(0)
    image = image.to(device)
    model = model.to(device)
    model.eval()
    with torch.inference_mode():
        preds = model(image)
    _, predicted_label = torch.max(preds, 1)
    return class_names[predicted_label.item()]

def main():
    model_path = pathlib.Path("models/") / "tiny_vgg.pth"
    # test_tinyvgg.jpg 04-pizza-dad.jpeg
    image_path = pathlib.Path("pytorch/") / "data" / "test_tinyvgg.jpg"
    class_names = ["pizza", "steak","sushi"]
    # model = model_builder.TinyVGG(
    #     input_shape=3,
    #     hidden_units=32,
    #     output_shape=3
    # )
    model ,auto_transforms= going_modular.transfer_learning.get_model()
    load_model(model_path, model)
    predicted_class = predict_image(image_path, model, class_names)
    print(f"Predicted class: {predicted_class}")

from typing import List, Tuple

from PIL import Image
from matplotlib import pyplot as plt
import torchvision

device = "mps" if torch.backends.mps.is_available() else "cuda" if torch.cuda.is_available() else "cpu"

# 1. Take in a trained model, class names, image path, image size, a transform and target device
def pred_and_plot_image(model: torch.nn.Module,
                        image_path: str, 
                        class_names: List[str],
                        image_size: Tuple[int, int] = (224, 224),
                        transform: torchvision.transforms = None,
                        device: torch.device=device):
    
    
    # 2. Open image
    img = Image.open(image_path)

    # 3. Create transformation for image (if one doesn't exist)
    if transform is not None:
        image_transform = transform
    else:
        image_transform = transforms.Compose([
            transforms.Resize(image_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                 std=[0.229, 0.224, 0.225]),
        ])

    ### Predict on image ### 

    # 4. Make sure the model is on the target device
    model.to(device)

    # 5. Turn on model evaluation mode and inference mode
    model.eval()
    with torch.inference_mode():
      # 6. Transform and add an extra dimension to image (model requires samples in [batch_size, color_channels, height, width])
      transformed_image = image_transform(img).unsqueeze(dim=0)

      # 7. Make a prediction on image with an extra dimension and send it to the target device
      target_image_pred = model(transformed_image.to(device))

    # 8. Convert logits -> prediction probabilities (using torch.softmax() for multi-class classification)
    target_image_pred_probs = torch.softmax(target_image_pred, dim=1)

    # 9. Convert prediction probabilities -> prediction labels
    target_image_pred_label = torch.argmax(target_image_pred_probs, dim=1)

    # 10. Plot image with predicted label and probability 
    plt.figure()
    plt.imshow(img)
    plt.title(f"Pred: {class_names[target_image_pred_label]} | Prob: {target_image_pred_probs.max():.3f}")
    plt.axis(False)
    plt.show()
    
if __name__ == "__main__":
    main()