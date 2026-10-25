import cv2
import numpy as np
img = cv2.imread("outputs/6_混合滤波.png")

if img is None:
    print("找不到 outputs/6_混合滤波.png，先运行 04_混合滤波.py")
    exit()
    
#转灰度
gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)

# Otsu：自动找阈值
thr,otsu = cv2.threshold(gray,0,255,cv2.THRESH_BINARY + cv2.THRESH_OTSU)
print("Otsu 自动计算阈值",thr)

# 固定阈值
_, fixed = cv2.threshold(gray,115,255,cv2.THRESH_BINARY)

# 自适应阈值：均值 / 高斯
adapt_mean = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY, 31, 8)
adapt_gauss = cv2.adaptiveThreshold(
    gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 8)

# 保存单张
cv2.imwrite("outputs/7_灰度.png", gray)
cv2.imwrite("outputs/7_Otsu.png", otsu)
cv2.imwrite("outputs/7_固定阈值.png", fixed)
cv2.imwrite("outputs/7_自适应均值.png", adapt_mean)
cv2.imwrite("outputs/7_自适应高斯.png", adapt_gauss)

# 拼成 2×2 对比图
top = np.hstack([otsu, fixed])
bottom = np.hstack([adapt_mean, adapt_gauss])
grid = np.vstack([top, bottom])
cv2.imwrite("outputs/7_二值化对比.png", grid)


print("完成：outputs/7_二值化对比.png")
