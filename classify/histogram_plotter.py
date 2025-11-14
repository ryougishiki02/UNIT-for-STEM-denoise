#    python histogram_plotter.py
import os
import numpy as np
import matplotlib.pyplot as plt
from fontTools.misc.classifyTools import classify
from scipy.stats import norm
from scipy.interpolate import make_interp_spline
from scipy.optimize import curve_fit
from matplotlib.widgets import RectangleSelector
import pandas as pd
from matplotlib.widgets import Cursor

#数据范围
a=0
b=300
#分割点
divide=125
# 读取数据
file_path = "classify/peak_heights.txt"
save_dir = "classify"
os.makedirs(save_dir, exist_ok=True)  # 确保目录存在
histogram_path = os.path.join(save_dir, "histogram_filtered.png")  # 直方图保存路径
curve_path = os.path.join(save_dir, "histogram_curve_smooth.png")  # 平滑曲线保存路径
gaussian_fit_path = os.path.join(save_dir, "gaussian_fit.png")  # 高斯拟合图保存路径
gaussian_fit_txt = os.path.join(save_dir, "gaussian_fits.txt")  # 高斯拟合信息保存路径
classification_txt = os.path.join(save_dir, "classification_counts.txt")  # 分类计数保存路径

# 解析数值数据
peak_heights = []
with open(file_path, "r", encoding="utf-8") as file:
    for line in file:
        parts = line.strip().split('\t')
        if len(parts) > 1:
            try:
                heights = list(map(float, parts[1].split()))
                peak_heights.extend(heights)
            except ValueError:
                continue

# 转换为 NumPy 数组
peak_heights = np.array(peak_heights)

# lower_bound = np.percentile(peak_heights, a)
# upper_bound = np.percentile(peak_heights, b)
filtered_heights = peak_heights[(peak_heights > a) & (peak_heights <= b)]

# **绘制直方图**
plt.figure(figsize=(10, 6))
hist_values, bin_edges, _ = plt.hist(filtered_heights, bins=250, edgecolor='black', alpha=0.7)
plt.title("Histogram of Peak Heights (Filtered)")
plt.xlabel("Peak Height")
plt.ylabel("Frequency")
plt.grid(True)
plt.savefig(histogram_path, dpi=300, bbox_inches='tight')
plt.show()
print(f"直方图已保存到 {histogram_path}")

# **计算 bin 的中心点**
bin_centers = (bin_edges[:-1] + bin_edges[1:]) / 2

# **三次样条插值，生成平滑曲线**
x_smooth = np.linspace(bin_centers.min(), bin_centers.max(), 300)
spline = make_interp_spline(bin_centers, hist_values, k=3)
y_smooth = spline(x_smooth)

# **绘制平滑曲线**
plt.figure(figsize=(10, 6))
plt.plot(x_smooth, y_smooth, linestyle='-', color='red', label="Smoothed Curve")
plt.scatter(bin_centers, hist_values, color='blue', s=10, label="Original Points")
plt.title("Peak Heights Distribution Curve (Smoothed)")
plt.xlabel("Peak Height")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.savefig(curve_path, dpi=300, bbox_inches='tight')
plt.show()
# 创建 DataFrame
df = pd.DataFrame({"x_smooth": x_smooth, "y_smooth": y_smooth})

# 保存为 TXT 文件（使用制表符 \t 分隔，适用于 Origin）
txt_path = "classify/'smoothed_curve.txt'"
df.to_csv(txt_path, sep="\t", index=False)
print(f"平滑曲线图已保存到 {curve_path}")

# **高斯函数定义**
def gaussian(x, a, x0, sigma):
    return a * np.exp(-((x - x0) ** 2) / (2 * sigma ** 2))


# **第一步：选择峰值**
print("请在图上左键点击要进行高斯拟合的峰值点，按 Enter 键结束选择。")

plt.figure(figsize=(10, 6))
plt.plot(x_smooth, y_smooth, linestyle='-', color='red', label="Smoothed Curve")
plt.title("Click on Peaks (Left Click), Press Enter to Finish")
plt.xlabel("X-axis")
plt.ylabel("Y-axis")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()

selected_peaks = plt.ginput(n=-1, timeout=0)  # 左键点击选择峰值
selected_x = [p[0] for p in selected_peaks]  # 获取峰值的 x 坐标
plt.close()

print(f"你选择的峰值位置: {selected_x}")


# **第二步：拖动选择拟合区域**
print("请在图上拖动鼠标选择拟合范围，松开鼠标即可确认。多次选择，按 'q' 退出。")

fig, ax = plt.subplots(figsize=(10, 6))
ax.plot(x_smooth, y_smooth, linestyle='-', color='red', label="Smoothed Curve")
ax.set_title("Drag to Select Fit Regions, Press 'q' to Finish")
ax.set_xlabel("X-axis")
ax.set_ylabel("Y-axis")
ax.grid(True, linestyle="--", alpha=0.6)
ax.legend()

fit_ranges = []

def onselect(eclick, erelease):
    """鼠标拖拽事件"""
    start, end = sorted([eclick.xdata, erelease.xdata])  # 确保 start < end
    fit_ranges.append((start, end))
    ax.axvspan(start, end, color='blue', alpha=0.3)  # 在图上标记选择的区域
    fig.canvas.draw_idle()  # 立即刷新图像

# **绑定拖拽选择器**
selector = RectangleSelector(ax, onselect, useblit=True, interactive=True)


# **退出方式**
def on_key(event):
    if event.key == 'enter':  # 按 'q' 退出
        plt.close()

fig.canvas.mpl_connect('key_press_event', on_key)
plt.show()

print(f"你选择的拟合区域: {fit_ranges}")


# **创建文件并写入标题**
with open(gaussian_fit_txt, "w", encoding="utf-8") as f:
    f.write("峰位置(x0)\t峰高(a)\t峰面积\t标准差(sigma)\t1σ范围\t2σ范围\t3σ范围\n")


# **进行高斯拟合**
plt.figure(figsize=(10, 6))
plt.plot(x_smooth, y_smooth, linestyle='-', color='gray', alpha=0.5, label="Smoothed Curve")

# for start, end in fit_ranges:
#     plt.axvspan(start, end, color='blue', alpha=0.3, label="Fit Range")

peaks_info = []

for peak_x in selected_x:
    # **按选定的范围筛选数据**
    for (range_start, range_end) in fit_ranges:
        if range_start <= peak_x <= range_end:  # 只在用户选定范围内进行拟合
            mask = (x_smooth >= range_start) & (x_smooth <= range_end)
            local_x = x_smooth[mask]
            local_y = y_smooth[mask]

            if len(local_x) < 5:
                print(f"⚠️ {range_start:.2f}-{range_end:.2f} 范围内数据不足，跳过！")
                continue

            # **强制使用用户选择的峰位置**
            idx = np.abs(local_x - peak_x).argmin()
            peak_y = local_y[idx]  # 保证峰值是用户点击的点
            p0 = [peak_y, peak_x, 10]  # 初始拟合参数

            try:
                popt, _ = curve_fit(gaussian, local_x, local_y, p0=p0)

                a_fit, x0_fit, sigma_fit = popt
                x0_fit = peak_x  # **强制 x0_fit 使用用户选择的峰位置**
                peak_area = a_fit * sigma_fit * np.sqrt(2 * np.pi)

                left_3sigma, right_3sigma = x0_fit - 3 * sigma_fit, x0_fit + 3 * sigma_fit

                peaks_info.append({
                    "x0": x0_fit,
                    "mean": x0_fit,  # 这里加上均值 mean
                    "sigma": sigma_fit,  # 添加标准差 sigma
                    "3sigma_range": (left_3sigma, right_3sigma)
                })

                # **写入文件**
                with open(gaussian_fit_txt, "a", encoding="utf-8") as f:
                    f.write(f"{x0_fit:.2f}\t{a_fit:.2f}\t{peak_area:.2f}\t{sigma_fit:.2f}\t"
                            f"[{left_3sigma:.2f}, {right_3sigma:.2f}]\n")

                # **绘制拟合曲线**
                gauss_x = np.linspace(local_x.min()-20, local_x.max()+20, 500)
                gauss_y = gaussian(gauss_x, *popt)
                plt.plot(gauss_x, gauss_y, linestyle='--', linewidth=2, label=f"Gaussian Fit @ {x0_fit:.2f}")

            except RuntimeError:
                print(f"⚠️ {range_start:.2f}-{range_end:.2f} 拟合失败，跳过！")
# **最终图形**
plt.title("Gaussian Fit on Selected Peaks")
plt.xlabel("Peak Height")
plt.ylabel("Frequency")
plt.grid(True, linestyle="--", alpha=0.6)
plt.legend()
plt.savefig(gaussian_fit_path, dpi=300, bbox_inches='tight')
plt.show()

print(f"高斯拟合图已保存到 {gaussian_fit_path}")
print(f"峰信息已保存到 {gaussian_fit_txt}")

# **分类计数**
atom_counts = {"A": 0, "B": 0, "C": 0,"all":0,"a_to_divide": 0, "divide_to_b": 0}

# **1. 读取峰信息**
first_left, first_right = peaks_info[0]["3sigma_range"]
second_left, second_right = peaks_info[1]["3sigma_range"]
first_sigma = peaks_info[0]["sigma"]
second_sigma = peaks_info[1]["sigma"]
first_mean = peaks_info[0]["mean"]
second_mean = peaks_info[1]["mean"]

# **2. 判断是否有交集**
has_intersection = first_right >= second_left  # 直接判断是否有交集

# **3. 计算交点（如果有交集）**
if has_intersection:
    def gaussian1(x):
        return norm.pdf(x, first_mean, first_sigma)

    def gaussian2(x):
        return norm.pdf(x, second_mean, second_sigma)

    # 在两个峰的范围内寻找交点
    x_values = np.linspace(first_right, second_left, 1000)
    y1 = gaussian1(x_values)
    y2 = gaussian2(x_values)
    diff = np.abs(y1 - y2)
    intersection_index = np.argmin(diff)
    intersection_x = x_values[intersection_index]  # 交点

# 读取并分类 peak_heights.txt
with open(file_path, "r", encoding="utf-8") as file:
    lines = file.readlines()[1:]  # **跳过表头**
    all_heights = []
    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) > 1:
            try:
                heights = list(map(float, parts[1].split()))  # **转换为浮点数**
                all_heights.extend(heights)
            except ValueError:
                continue  # **如果无法转换，跳过该行**
            filtered_heights = [h for h in all_heights if a <= h <= b]
            for height in filtered_heights:
                if has_intersection:
                    # **有交集时，按交点分类**
                    if height <= intersection_x:
                        atom_counts["A"] += 1
                    else:
                        atom_counts["B"] += 1
                else:
                    in_A = first_left <= height <= first_right
                    in_B = second_left <= height <= second_right
                    if in_A:
                        atom_counts["A"] += 1
                    elif in_B:
                        atom_counts["B"] += 1
                    elif height < first_left:
                        atom_counts["A"] += 1
                    elif first_right < height < second_left:
                        atom_counts["C"] += 1
                    elif height > second_right:
                        atom_counts["B"] += 1
                atom_counts["all"] = atom_counts["A"] + atom_counts["B"]  + atom_counts["C"]
                if height <= divide:
                    atom_counts["a_to_divide"] += 1
                else:
                    atom_counts["divide_to_b"] += 1

with open(classification_txt, "w", encoding="utf-8") as f:
    f.write("有交集\n" if has_intersection else "无交集\n")  # 记录交集信息
    if has_intersection:
        f.write(f"交点位置: {intersection_x:.2f}\n")  # 添加交点位置

    for atom, count in atom_counts.items():
        f.write(f"{atom}\t{count}\n")

    # 保存 a 到 divide 和 divide 到 b 的具体范围和计数
    f.write(f"\n从 {a} 到 {divide} 的计数: {atom_counts['a_to_divide']}\n")
    f.write(f"从 {divide} 到 {b} 的计数: {atom_counts['divide_to_b']}\n")

print(f"分类计数结果已保存到 {classification_txt}")










