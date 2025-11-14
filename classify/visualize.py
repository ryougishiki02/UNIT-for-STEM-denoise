import cv2
import time
import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from scipy.interpolate import griddata
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
from matplotlib import colors
from matplotlib.widgets import RectangleSelector
from scipy.ndimage import gaussian_filter
from scipy.optimize import curve_fit
from skimage.feature import peak_local_max
from scipy.spatial import distance
from scipy.optimize import minimize
import matplotlib.cm as cm
from concurrent.futures import ProcessPoolExecutor
from scipy.spatial import KDTree
from sklearn.cluster import KMeans
from collections import Counter
from scipy.signal import find_peaks
import random

def display_atom_positions(frame, positions, markersize=5, save_path=None):
    plt.figure(figsize=(10.24, 10.24),dpi=100)
    plt.imshow(frame, cmap='gray',alpha=1)
    plt.axis('off')  # 隐藏坐标轴
    # 使用 scatter 一次性绘制
    x_vals, y_vals = zip(*positions)  # 解压坐标列表
    plt.scatter(x_vals, y_vals, color='red', s=markersize ** 2)
    if save_path:
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0, dpi=300)
    plt.show()