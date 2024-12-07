import requests
import zipfile
from pathlib import Path
import os

data_path = Path('data/')
image_path = data_path / "pizza_steak_sushi"

def review_datafile():
    """
    https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip
    检查指定的图像目录是否存在，如果不存在，则创建它。
    从指定的URL下载包含披萨、牛排和寿司图像的zip文件。
    将下载的文件解压到图像目录中。
    异常:
        requests.exceptions.RequestException: 如果HTTP请求有问题。
        zipfile.BadZipFile: 如果下载的文件不是zip文件或已损坏。
    """
    if image_path.is_dir():
        print(f"{image_path} directory exists.")
    else:
        print(f"Did not find {image_path} directory, creating one ...")
        image_path.mkdir(parents=True, exist_ok=True)
    
    with open(data_path / "pizza_steak_sushi.zip", "wb") as f:
        request = requests.get("https://github.com/mrdbourke/pytorch-deep-learning/raw/main/data/pizza_steak_sushi.zip")
        print("Downloading pizza, steak , sushi data ...")
        f.write(request.content)
    
    with zipfile.ZipFile(data_path / "pizza_steak_sushi.zip", "r") as zip_ref:
        print("Unzipping pizza, steak, sushi data ...")
        zip_ref.extractall(image_path)
    
    os.remove(data_path / "pizza_steak_sushi.zip")

if "__main__" == __name__:
    review_datafile()