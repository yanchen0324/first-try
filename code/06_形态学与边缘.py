import cv2
import numpy as np

otsu = cv2.imread("outputs/7_Otsu.png", cv2.IMREAD_GRAYSCALE)
gray = cv2.imread("outputs/7_灰度.png", cv2.IMREAD_GRAYSCALE)

if otsu is None or gray is None:
    print("找不到 7_Otsu.png / 7_灰度.png，先运行 05_二值化.py")
    exit()

# 反色：让人变成白色前景
person = cv2.bitwise_not(otsu)

# 7×7 椭圆核
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))

eroded = cv2.erode(person, kernel, iterations=1)
dilated = cv2.dilate(person, kernel, iterations=1)

# Canny：高阈值改回 160，边缘更干净
edges = cv2.Canny(gray, 60, 160)

# Sobel
gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
mag = cv2.magnitude(gx, gy)
sobel = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

# 保存
cv2.imwrite("outputs/8_前景.png", person)
cv2.imwrite("outputs/8_腐蚀.png", eroded)
cv2.imwrite("outputs/8_膨胀.png", dilated)
cv2.imwrite("outputs/8_Canny.png", edges)
cv2.imwrite("outputs/8_Sobel.png", sobel)

# 拼 2×3
row1 = np.hstack([gray, person, eroded])
row2 = np.hstack([dilated, edges, sobel])
grid = np.vstack([row1, row2])
cv2.imwrite("outputs/8_形态边缘对比.png", grid)

print("完成！")
