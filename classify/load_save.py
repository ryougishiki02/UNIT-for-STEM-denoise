import cv2
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import RectangleSelector
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from scipy.spatial import distance
from scipy.optimize import minimize
import matplotlib.cm as cm
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import KDTree
import PIL
from tkinter import filedialog
from PIL import Image, ImageTk

def load_img(img_path, size=None, mode=None, crop_square=False, show=False):
    """
    加载图像，支持调整大小、颜色模式转换，并可裁剪为正方形。

    参数:
    img_path (str): 图像文件路径。
    size (tuple, 可选): 调整后的图像大小 (宽, 高)。
    mode (str, 可选): 转换图像的颜色模式（如 "L" 为灰度）。
    crop_square (bool, 可选): 是否裁剪图像为正方形，默认 False。

    返回:
    Image: 图像对象。
    """
    img = Image.open(img_path)

    # 如果需要裁剪为正方形
    if crop_square:
        width, height = img.size
        min_dim = min(width, height)
        left = (width - min_dim) // 2
        top = (height - min_dim) // 2
        right = left + min_dim
        bottom = top + min_dim
        img = img.crop((left, top, right, bottom))  # 裁剪为中心正方形

    # 调整大小
    if size:
        img = img.resize(size)

    # 转换颜色模式
    if mode:
        img = img.convert(mode)

    if show:
        img.show()

    return img

# 加载视频文件
def load_video(file_path):
    cap = cv2.VideoCapture(file_path)
    frame_rate = cap.get(cv2.CAP_PROP_FPS)
    frames = []
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames, frame_rate

# 保存视频
def save_video(frames, output_path, frame_rate):
    height, width, layers = frames[0].shape
    video = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), frame_rate, (width, height))
    for frame in frames:
        video.write(frame)
    video.release()

def open_image_file():
    file_path = filedialog.askopenfilename(filetypes=[("Image Files", "*.png;*.jpg;*.jpeg;*.bmp;*.gif;*.tif")])

    if file_path:
        # 打开图像并返回图像对象
        image = Image.open(file_path)
        return image
    return None