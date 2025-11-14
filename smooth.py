#     python smooth.py

import cv2
import numpy as np
import os


def smooth_image_edges(image, blur_size=5, mask_strength=50):
    """
    对图像边界进行平滑处理，减少拼接伪影。
    :param image: 输入图像 (numpy 数组)
    :param blur_size: 平滑模糊核大小，值越大，模糊越强
    :param mask_strength: 控制边缘平滑过渡的强度，值越大，边缘模糊过渡更缓慢
    :return: 处理后的平滑图像
    """
    h, w = image.shape[:2]

    # 生成权重掩码，使边缘区域平滑过渡
    x = np.linspace(0, 1, w).reshape(1, -1) ** mask_strength
    y = np.linspace(0, 1, h).reshape(-1, 1) ** mask_strength
    mask = np.minimum(np.minimum(x, 1 - x), np.minimum(y, 1 - y))
    mask = np.expand_dims(mask, axis=-1)  # 适配通道数

    # 应用高斯模糊
    blurred = cv2.GaussianBlur(image, (blur_size, blur_size), 0)

    # 使用掩码融合原图和模糊图
    smooth_image = (image * mask + blurred * (1 - mask)).astype(np.uint8)

    return smooth_image


# 处理目录下的所有图像
def process_images(input_dir, output_dir):
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    for filename in os.listdir(input_dir):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, filename)

        image = cv2.imread(input_path)
        if image is not None:
            smooth_img = smooth_image_edges(image)
            cv2.imwrite(output_path, smooth_img)
            print(f"Processed: {filename}")
        else:
            print(f"Skipping: {filename} (not a valid image)")


# 设置输入和输出文件夹
input_folder = "smooth/inputs"
output_folder = "smooth/outputs"
process_images(input_folder, output_folder)
