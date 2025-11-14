#    python SSIM_FSIM.py
import cv2
import numpy as np
import os
from sklearn.metrics import mean_squared_error


a=1
b=1

# 设定路径
label_path = "Metric/5/label.png"  # 参考图像
result_dir = "Metric/5/batch_results/"  # 结果文件夹
output_file ="Metric/5/results.txt"  # 输出文件

def calculate_psnr(img1, img2):
    """计算 PSNR"""
    # 检查图像的数据类型和最大像素值
    # print(f"img1 dtype: {img1.dtype}, max value: {np.max(img1)}")
    # print(f"img2 dtype: {img2.dtype}, max value: {np.max(img2)}")
    # print(type(img1))  # 应该输出 <class 'numpy.ndarray'>
    # print(type(img2))  # 应该输出 <class 'numpy.ndarray'>

    # mse = np.mean((img1 - img2) ** 2)
    # mse = np.mean(np.square(img1 - img2))
    mse = mean_squared_error(img1, img2)

    if mse == 0:
        return float('inf')  # 避免 log(0) 错误
    max_pixel = 255  # 8-bit 图像
    print(mse)
    return 10 * np.log10((max_pixel ** 2) / mse)


# def calculate_psnr(img1, img2):
#     """计算 PSNR（使用 float32 避免溢出）"""
#     # 确保图像是 np.float32 类型
#     img1 = img1.astype(np.float32)
#     img2 = img2.astype(np.float32)
#
#     # 计算均方误差（MSE）
#     mse = np.mean((img1 - img2) ** 2, dtype=np.float32)  # 确保用 float32 计算
#
#     # 避免 log(0) 计算错误
#     if mse == 0:
#         return float('inf')
#     print(mse)
#
#     # 计算 PSNR
#     max_pixel = np.max([img1.max(), img2.max()])  # 动态确定最大像素值
#     psnr = 10 * np.log10((max_pixel ** 2) / mse)
#
#     return psnr



# def calculate_ssim(img1, img2):
#     """计算 SSIM"""
#     C1 = (0.01 * 255) ** 2
#     C2 = (0.03 * 255) ** 2
#
#     mu_x = cv2.GaussianBlur(img1, (a, a), b)
#     mu_y = cv2.GaussianBlur(img2, (a, a), b)
#
#     sigma_x2 = cv2.GaussianBlur(img1 ** 2, (a, a), b) - mu_x ** 2
#     sigma_y2 = cv2.GaussianBlur(img2 ** 2, (a, a), b) - mu_y ** 2
#     sigma_xy = cv2.GaussianBlur(img1 * img2, (a, a), b) - mu_x * mu_y
#
#     ssim_value = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / (
#             (mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2))
#
#     return np.mean(ssim_value)

def calculate_ssim(img1, img2):
    """计算 SSIM (结构相似性)"""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # 确保 C1 和 C2 适配图像范围
    L = np.max([img1.max(), img2.max()])  # 自动检测最大像素值
    C1 = (0.01 * L) ** 2
    C2 = (0.03 * L) ** 2

    # 计算均值
    mu_x = cv2.GaussianBlur(img1, (a, a), b)
    mu_y = cv2.GaussianBlur(img2, (a, a), b)

    # 计算方差
    sigma_x2 = cv2.GaussianBlur(img1 ** 2, (a, a), b) - mu_x ** 2
    sigma_y2 = cv2.GaussianBlur(img2 ** 2, (a, a), b) - mu_y ** 2
    sigma_xy = cv2.GaussianBlur(img1 * img2, (a, a), b) - mu_x * mu_y

    # 避免负值
    sigma_x2 = np.maximum(sigma_x2, 0)
    sigma_y2 = np.maximum(sigma_y2, 0)

    # 计算 SSIM
    ssim_value = ((2 * mu_x * mu_y + C1) * (2 * sigma_xy + C2)) / \
                 ((mu_x ** 2 + mu_y ** 2 + C1) * (sigma_x2 + sigma_y2 + C2))

    return np.mean(ssim_value)  # 返回均值 SSIM

def phase_congruency(image):
    """计算相位一致性（PC）"""
    dx = cv2.Sobel(image, cv2.CV_64F, 1, 0, ksize=3)
    dy = cv2.Sobel(image, cv2.CV_64F, 0, 1, ksize=3)
    gradient_magnitude = np.sqrt(dx**2 + dy**2)

    # 归一化梯度幅度到 [0,1]
    normalized_gm = (gradient_magnitude - np.min(gradient_magnitude)) / \
                    (np.max(gradient_magnitude) - np.min(gradient_magnitude) + 1e-6)
    return normalized_gm

def calculate_fsim(img1, img2, alpha=0.5, beta=0.5):
    """计算 FSIM 指标"""
    img1 = img1.astype(np.float32)
    img2 = img2.astype(np.float32)

    # 计算 PC（相位一致性）
    PC1 = phase_congruency(img1)
    PC2 = phase_congruency(img2)
    PCm = np.maximum(PC1, PC2)  # 取最大值

    # 计算梯度幅度
    G1_x = cv2.Sobel(img1, cv2.CV_64F, 1, 0, ksize=3)
    G1_y = cv2.Sobel(img1, cv2.CV_64F, 0, 1, ksize=3)
    G1 = np.sqrt(G1_x**2 + G1_y**2)

    G2_x = cv2.Sobel(img2, cv2.CV_64F, 1, 0, ksize=3)
    G2_y = cv2.Sobel(img2, cv2.CV_64F, 0, 1, ksize=3)
    G2 = np.sqrt(G2_x**2 + G2_y**2)

    Gm = np.maximum(G1, G2)  # 取最大梯度

    # 归一化 Gm 到 [0,1]
    Gm = (Gm - np.min(Gm)) / (np.max(Gm) - np.min(Gm) + 1e-6)

    # 计算 FSIM
    FSIM_map = (PCm ** alpha) * (Gm ** beta)

    # 归一化 FSIM_map，确保值在 [0,1]
    FSIM_score = np.sum(FSIM_map) / (np.sum(PCm) + 1e-6)
    FSIM_score = max(0, min(1, FSIM_score))  # 确保 FSIM 在 [0,1] 范围内
    return FSIM_score




# 读取 label 图像并转换为灰度图（保存修改后的灰度版本）
label_img = cv2.imread(label_path, cv2.IMREAD_GRAYSCALE)
if label_img is None:
    print(f"❌ 无法读取参考图像: {label_path}，请检查文件路径！")
    exit()

cv2.imwrite(label_path, label_img)  # 重新保存，确保文件是灰度的

# 遍历目录，统一转换为灰度 PNG 格式，并删除原 JPG/PNG
results = []
for filename in os.listdir(result_dir):
    file_path = os.path.join(result_dir, filename)

    # 读取待测图片
    test_img = cv2.imread(file_path, cv2.IMREAD_GRAYSCALE)
    if test_img is None:
        print(f"⚠️ 无法读取图像: {filename}，跳过")
        continue

    # 统一转换为 PNG 并删除原 JPG/PNG
    new_filename = os.path.splitext(filename)[0] + ".png"
    new_path = os.path.join(result_dir, new_filename)

    cv2.imwrite(new_path, test_img)  # 保存为灰度 PNG
    if file_path != new_path:  # 如果原文件不是新文件（说明是 JPG 或 PNG）
        os.remove(file_path)  # 删除原文件
        print(f"✅ {filename} 转换为灰度 {new_filename} 并删除原文件")

    # 计算 PSNR 和 SSIM
    psnr_value = calculate_psnr(label_img, test_img)
    ssim_value = calculate_ssim(label_img.astype(np.float32), test_img.astype(np.float32))
    fsim_value = calculate_fsim(label_img, test_img)

    # 存储结果
    results.append(f"{new_filename}: PSNR={psnr_value:.2f}, SSIM={ssim_value:.3f}, FSIM={fsim_value:.3f}")

# 保存到文本文件
with open(output_file, "w") as f:
    f.write("\n".join(results))

print(f"\n✅ 计算完成，结果已保存至 {output_file}")





