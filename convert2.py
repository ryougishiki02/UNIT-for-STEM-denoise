#python convert2.py
from PIL import Image
import os

# 输入和输出文件夹路径（同一文件夹）
input_folder = r'datasets\STM\testA'  # 替换为你的图片文件夹路径

# 遍历文件夹中的所有 .jpg 图片
for filename in os.listdir(input_folder):
    if filename.endswith(".jpg"):  # 仅处理 .jpg 格式
        img_path = os.path.join(input_folder, filename)

        # 打开图像并转换为灰度
        img = Image.open(img_path).convert("L")  # 转换为单通道灰度图像

        # 覆盖保存
        img.save(img_path)

print("所有图片已转换为单通道灰度图像并覆盖保存。")
