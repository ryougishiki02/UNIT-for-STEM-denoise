
#    python divide.py

import os
from PIL import Image

# 输入和输出路径
input_folder = 'divide\input'  # 替换为实际文件夹路径
output_path = 'divide\output'  # 替换为实际输出文件夹路径
n = 5  # 你想将图片切成 n x n 小图
pixel_increase = 50

# 检查输出路径是否存在，如果不存在则创建
if not os.path.exists(output_path):
    os.makedirs(output_path)

# 获取输入文件夹中的所有文件
file_list = os.listdir(input_folder)

# 筛选出所有图片文件（可以根据需要添加其他图片格式）
image_extensions = ['.jpg', '.jpeg', '.png', '.tif', '.bmp']
image_files = [f for f in file_list if any(f.lower().endswith(ext) for ext in image_extensions)]

# 遍历文件夹中的每个图片文件
for image_name in image_files:
    # 获取图片的完整路径
    input_path = os.path.join(input_folder, image_name)

    try:
        # 打开图片
        print(f"Processing image: {input_path}")
        img = Image.open(input_path)

        # # 如果图片是调色板模式（P），转换为 RGB 模式
        # if img.mode == 'P':
        #     img = img.convert('RGB')

        # 确保图片为 RGB 模式
        if img.mode != 'RGB':
            img = img.convert('RGB')

        # 获取图片的宽高
        width, height = img.size
        print(f"Image size: {width}x{height}")

        # 检查是否为正方形图片
        if width != height:
            print(f"Skipping {image_name} because it is not a square image.")
            continue

        # 计算每个小图的尺寸
        sub_width = width // n
        sub_height = height // n

        # 计算扩展后的每个小图的尺寸
        expanded_sub_width = sub_width + pixel_increase
        expanded_sub_height = sub_height + pixel_increase

        # 按照 n 切割图片，带有重叠区域
        for i in range(n):
            for j in range(n):
                # 计算每个小图的裁剪区域
                if j == n - 1:  # 最后一列，调整裁剪区域
                    left = j * sub_width - pixel_increase
                else:
                    left = j * sub_width - pixel_increase // 2

                if i == n - 1:  # 最后一行，调整裁剪区域
                    upper = i * sub_height - pixel_increase
                else:
                    upper = i * sub_height - pixel_increase // 2

                # 确保裁剪区域不超出图片边界
                left = max(left, 0)
                upper = max(upper, 0)

                # 计算右下角的坐标
                right = left + expanded_sub_width
                lower = upper + expanded_sub_height


                # right = min(right, width)
                # lower = min(lower, height)

                # 裁剪出小图
                sub_img = img.crop((left, upper, right, lower))

                # 调整小图分辨率为 256x256 并保持为 RGB
                sub_img = sub_img.resize((256, 256))

                # 去掉原文件扩展名，以避免命名重复
                base_name = os.path.splitext(image_name)[0]
                # 生成文件名，例如 '11', '12', '21', '22' 等
                file_name = f"{i + 1}{j + 1}_{base_name}.jpg"

                # 保存为三通道RGB图片
                sub_img.save(os.path.join(output_path, file_name), 'JPEG')

        print(f"Images from {image_name} saved to {output_path}")

    except Exception as e:
        print(f"An error occurred while processing {image_name}: {e}")

