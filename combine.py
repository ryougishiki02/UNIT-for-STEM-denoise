

#   python combine.py

import os
from PIL import Image
from collections import defaultdict
from divide import pixel_increase,expanded_sub_width,expanded_sub_height
import numpy as np
import cv2


def merge_images(image_folder, output_folder):
    # 获取文件夹中的所有图片文件
    files = [f for f in os.listdir(image_folder) if f.endswith('.jpg')]

    # # 强制将文件名解码为 UTF-8
    #files = [f.encode('utf-8').decode('utf-8') for f in files]

    # 按原图名分组：假设文件名格式为 '12_0007-FFTanddelete-S.jpg'
    groups = defaultdict(list)
    for file in files:
        base_name = file.split('_', 1)[1].split('.jpg')[0]  # 取第二部分并去除后缀
        groups[base_name].append(file)  # 将文件按原图名分组

    # 处理每一组图片
    for group_name, group_files in groups.items():
        print(f"正在处理组: {group_name}")

        # 提取行和列的最大值
        max_row = 0
        max_col = 0
        for file in group_files:
            # 提取文件名中的前两个或四个数字（行列信息）
            prefix_str = file.split('_')[0]  # 获取前两个或四个数字

            # 判断是两个数字还是四个数字
            if len(prefix_str) == 2:
                row, col = map(int, prefix_str)  # 如果是两个数字
            elif len(prefix_str) == 4:
                row, col = map(int, [prefix_str[:2], prefix_str[2:]])  # 如果是四个数字
            max_row = max(max_row, row)  # 获取最大的行数
            max_col = max(max_col, col)  # 获取最大的列数

        # 拼接后大图的尺寸：行数 * 图片宽度, 列数 * 图片高度
        n = max(max_row, max_col)  # 选择最大的行列数
        sample_image = Image.open(os.path.join(image_folder, group_files[0]))
        img_width, img_height = sample_image.size
        large_img_width = (expanded_sub_width - pixel_increase) * n
        large_img_height = (expanded_sub_height - pixel_increase) * n

        # 创建一个空白大图和计数器图，用于记录覆盖次数
        large_image = np.zeros((large_img_height, large_img_width, 3), dtype=np.float32)
        count_image = np.zeros((large_img_height, large_img_width), dtype=np.int32)

        # 遍历小图片，将它们放到大图相应的位置
        for file in group_files:
            # 获取当前图片的行列位置
            prefix_str = file.split('_')[0]
            if len(prefix_str) == 2:
                row, col = map(int, prefix_str)  # 如果是两个数字
            elif len(prefix_str) == 4:
                row, col = map(int, [prefix_str[:2], prefix_str[2:]])  # 如果是四个数字

            # 计算该图片在大图中的位置
            if row == n:
                upper = (row - 1) * (expanded_sub_height - pixel_increase) - pixel_increase
            else:
                upper = (row - 1) * (expanded_sub_height - pixel_increase) - pixel_increase // 2
            if col == n:
                left = (col - 1) * (expanded_sub_width - pixel_increase) - pixel_increase
            else:
                left = (col - 1) * (expanded_sub_width - pixel_increase) - pixel_increase // 2

            # 确保裁剪区域不超出图片边界
            left = max(left, 0)
            upper = max(upper, 0)

            # 读取小图片
            small_image = cv2.imread(os.path.join(image_folder, file))  # 确保路径正确

            # 检查图片是否读取成功
            if small_image is None:
                print(f"Failed to load image: {file}")
            else:
                # 调整小图的尺寸
                small_image = cv2.resize(small_image, (expanded_sub_width, expanded_sub_height))

                # 获取调整后的图像尺寸
                small_img_height, small_img_width, _ = small_image.shape

                # 计算裁剪区域
                lower = upper + small_img_height
                right = left + small_img_width

                # 计算裁剪部分
                small_img_crop = small_image[:lower - upper, :right - left]

                # 将小图覆盖到大图中，同时更新计数器
                large_image[upper:lower, left:right] += small_img_crop
                count_image[upper:lower, left:right] += 1

        # 使用计数器对重叠区域进行平均
        # 防止除以零
        count_image = np.where(count_image == 0, 1, count_image)  # 将 count_image 中的 0 替换为 1，避免除以零错误
        large_image /= count_image[..., None]  # 对每个通道进行平均

        # 将结果转换为整数，并转回图像格式
        large_image = np.clip(large_image, 0, 255).astype(np.uint8)
        large_image = Image.fromarray(large_image)

        # 获取原始大图的名称部分（去掉前缀的四个数字）
        original_image_name = group_files[0].split('_', 1)[1].split('.jpg')[0]

        # 最终文件名：n-new-原图名.jpg
        output_filename = f"{n}-{pixel_increase}-{original_image_name}.jpg"

        output_path = os.path.join(output_folder, output_filename)

        # 保存合并后的大图
        large_image.save(output_path)
        print(f"大图已保存为: {output_path}")


# 调用示例：
image_folder = 'combine/inputs'  # 小图所在的文件夹
output_folder = 'combine/outputs'  # 保存大图的文件夹

merge_images(image_folder, output_folder)
