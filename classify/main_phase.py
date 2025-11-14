#   python main_phase.py

import numpy as np

import os
import pkg, load_save, visualize

# 输入和输出文件夹
image_folder = 'classify/inputs'
output_folder = 'classify/outputs'
peak_heights_file = 'classify/peak_heights.txt'  # 设定峰高保存的文件路径

# 确保输出文件夹存在
os.makedirs(output_folder, exist_ok=True)

# 允许的图片扩展名（增加 .tif 和 .tiff）
valid_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.gif', '.tif', '.tiff')

# 获取文件夹内的所有图片文件
image_files = [f for f in os.listdir(image_folder) if f.lower().endswith(valid_extensions)]

# 确保文件夹不为空
if not image_files:
    print("Error: No image files found in the folder!")
else:
    # 打开txt文件进行保存峰高数据
    with open(peak_heights_file, mode='w') as file:
        # 写入表头
        file.write("Image File\tPeak Heights\n")

        for image_file in image_files:
            image_path = os.path.join(image_folder, image_file)  # 生成完整路径
            print(f"Processing image: {image_path}")  # 打印处理中的图片

            # 加载图片（支持 .tif）
            image = load_save.load_img(image_path, mode='L', show=False)

            # 进行图像处理
            positions,peak_heights  = pkg.process_frame(image, sigma=1, min_distance=4, threshold_abs=20, cut_size=10,
                                                        visualize=True)

            # 将峰高值四舍五入到两位小数
            peak_heights_rounded = [round(height, 2) for height in peak_heights]

            # 将峰高保存到文件
            peak_heights_str = " ".join(map(str, peak_heights_rounded))  # 将四舍五入后的峰高值转换为字符串并用空格隔开
            file.write(f"{image_file}\t{peak_heights_str}\n")  # 每行写入图像文件名和峰高值

            # 生成与输入文件同名的输出路径
            save_path = os.path.join(output_folder, image_file)

            # 结果可视化并保存
            visual = [1, 0, 0]  # 是否可视化（第一个参数=1 代表开启可视化）
            if visual[0]:
                visualize.display_atom_positions(image, positions, markersize=1, save_path=save_path)
                print(f"Saved: {save_path}")

            print(f"Peak heights saved for {image_file}")

        print(f"Peak heights have been saved to {peak_heights_file}")

