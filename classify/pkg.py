import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from scipy.spatial import KDTree
from skimage.feature import peak_local_max
from sklearn.cluster import DBSCAN
from scipy.stats import gaussian_kde
import tkinter as tk
from tkinter import ttk, filedialog
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg


# 定义简化的二维高斯函数
def gaussian2d(xy, A, x0, y0, sigma_x, sigma_y, B):
    x, y = xy
    return A * np.exp(-((x - x0) ** 2 / (2 * sigma_x ** 2) + (y - y0) ** 2 / (2 * sigma_y ** 2))) + B
def filter_close_positions(positions, min_distance, filtered_img):
    """
    使用KDTree加速距离计算，过滤掉距离小于min_distance的坐标，保留像素值最大的那个。
    参数：
    positions: 原子坐标列表，每个坐标为[行, 列]。
    min_distance: 距离阈值，距离小于该值的坐标会被过滤掉。
    filtered_img: 图像数据，用于比较每个位置的像素值。
    返回：
    updated_positions: 筛选后的坐标列表。
    """
    # 创建KDTree
    tree = KDTree(positions)
    updated_positions = []

    # 遍历每个坐标
    for i, (x, y) in enumerate(positions):
        # 当前坐标的像素值
        value = filtered_img[int(y), int(x)]

        # 查找距离当前点小于min_distance的所有点
        neighbors = tree.query_ball_point([x, y], min_distance)

        # 如果没有邻居，则直接添加
        if not neighbors:
            updated_positions.append([x, y])
        else:
            # 检查邻居中像素值最大的点
            max_value = value
            best_pos = [x, y]
            for idx in neighbors:
                xx, yy = positions[idx]
                neighbor_value = filtered_img[int(yy), int(xx)]
                if neighbor_value > max_value:
                    max_value = neighbor_value
                    best_pos = [xx, yy]

            # 添加最优位置
            updated_positions.append(best_pos)

    return updated_positions

def process_frame(frame, roi=None, sigma=2, min_distance=5, threshold_abs=100, cut_size=10, visualize=False):
    #     gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray_frame = frame

    # 对整张图进行高斯滤波
    filtered_img = gaussian_filter(gray_frame, sigma=sigma)
    coordinates = peak_local_max(filtered_img, min_distance, threshold_abs, footprint=np.ones((3, 3)))

    fit_results = []
    peak_heights = []  # 用来保存峰的高度
    for (y, x) in coordinates:
        x0, x1 = x - cut_size, x + cut_size + 1
        y0, y1 = y - cut_size, y + cut_size + 1
        if x0 < 0 or x1 > filtered_img.shape[1] or y0 < 0 or y1 > filtered_img.shape[0]:
            continue
        sub_img = filtered_img[y0:y1, x0:x1]
        x_grid, y_grid = np.meshgrid(np.arange(x0, x1), np.arange(y0, y1))
        xdata = np.vstack((x_grid.ravel(), y_grid.ravel()))
        initial_guess = (sub_img.max(), x, y, 0.05,0.05, sub_img.min())
        try:
            popt, _ = curve_fit(gaussian2d, xdata, sub_img.ravel(), p0=initial_guess)
            if popt[1] < 0 or popt[1] > filtered_img.shape[1] or popt[2] < 0 or popt[2] > filtered_img.shape[0]:
                popt[1] = x
                popt[2] = y
            fit_results.append(popt)
            peak_heights.append(popt[0])  # 保存每个峰的高度（振幅A）
        except RuntimeError:
            continue

    positions = np.array([[fit[1], fit[2]] for fit in fit_results], dtype=np.float32)
    positions = np.array(filter_close_positions(positions, 8, filtered_img))
    print(len(positions))

    # 如果有设置ROI，筛选出位置在ROI内的原子
    if roi is not None:
        x, y, w, h = roi
        positions = np.array([pos for pos in positions if x <= pos[0] <= x + w and y <= pos[1] <= y + h],
                             dtype=np.float32)
    for x, y in positions:
        if x < 0 or x > gray_frame.width or y < 0 or y > gray_frame.height:
            positions = np.delete(positions, np.where(positions == [x, y])[0], axis=0)

    # 可视化
    if visualize:
        for i in [1, 7, 20, 40, 50]:
            x, y = positions[i]
        # 确定裁剪区域的边界
        crop_size = 5 * cut_size
        left = max(0, int(x - crop_size))
        upper = max(0, int(y - crop_size))
        right = min(gray_frame.width, int(x + crop_size + 1))
        lower = min(gray_frame.height, int(y + crop_size + 1))

        # 使用 PIL 的 crop 方法裁剪区域
        crop_frame = gray_frame.crop((left, upper, right, lower))

        crop_x, crop_y = crop_frame.size

        plt.figure(figsize=(10, 10))
        plt.imshow(crop_frame, cmap='gray')

        # 绘制圆形表示 min_distance
        circle = plt.Circle((crop_x/2, crop_y/2), radius=min_distance, color='red', fill=False, linestyle='-', linewidth=1.5)
        plt.gca().add_patch(circle)

        # 绘制方形表示 cut_size
        rect_x = crop_x/2 - cut_size
        rect_y = crop_y/2 - cut_size
        square = plt.Rectangle((rect_x, rect_y), 2 * cut_size + 1, 2 * cut_size + 1, edgecolor='blue', fill=False, linestyle='-', linewidth=1.5)
        plt.gca().add_patch(square)

        plt.title("Detected Points with min_distance and cut_size Visualization")
        plt.colorbar()
        plt.show()

    return positions, peak_heights

# def frame_to_square(frames):
#     frame = frames[0]
#     # 获取帧的高度和宽度
#     height, width = frame.shape[:2]
#     size = min(height, width)
#     new_frames = []
#
#     y = int((height - size) / 2)
#     x = int((width - size) / 2)
#     for frame in frames:
#         # 转换为灰度图
#         if len(frame.shape) == 3 and frame.shape[2] == 3:
#             gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#         else:
#             # 如果输入不是预期的彩色图像形状，抛出异常或处理错误
#             raise ValueError("Expected a color image with 3 channels, got shape: {}".format(frame.shape))
#
#         new_frame = gray_frame[y:y + size, x:x + size]
#         new_frames.append(new_frame)
#
#     return new_frames
