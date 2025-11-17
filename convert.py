
#python convert.py
from PIL import Image
import os

# 输入和输出文件夹路径
input_folder = 'datasets\STEM/testB'  # 替换为你的图片文件夹路径
#input_folder = 'inputs'
output_folder = 'datasets\STEM/testB'  # 设置转换后图片的保存路径
#output_folder = 'inputs'

# 如果输出文件夹不存在，则创建
os.makedirs(output_folder, exist_ok=True)

# 批量转换 PNG 图片为 RGB 格式
for filename in os.listdir(input_folder):
    if filename.endswith('.png'):  # 仅处理 PNG 格式的图片
        image_path = os.path.join(input_folder, filename)
        img = Image.open(image_path)
        img_rgb = img.convert('RGB')  # 转换为 RGB 格式
        # 将文件扩展名更改为 .jpg
        output_filename = os.path.splitext(filename)[0] + '.jpg'
        output_path = os.path.join(output_folder, output_filename)

        # 保存为 JPG 格式
        img_rgb.save(output_path, 'JPEG')
        print(f"Converted {filename} to RGB and saved as {output_path}")
        # 删除原始 PNG 文件
        os.remove(image_path)
        print(f"Deleted original PNG file: {image_path}")
