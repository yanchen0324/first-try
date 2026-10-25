import cv2

img = cv2.imread("图1.png")

# 归一化方框（= 均值）
box_norm = cv2.boxFilter(img, -1, (5, 5), normalize=True)

# 不归一化方框（直接求和）
box_raw = cv2.boxFilter(img, -1, (5, 5), normalize=False)

# 均值
mean = cv2.blur(img, (5, 5))

# 算最大像素差
diff = cv2.absdiff(box_norm, mean)
print("归一化方框 vs 均值，最大像素差:", diff.max())
print("不归一化方框的平均亮度:", box_raw.mean())

cv2.imwrite("outputs/5_方框_不归一化.png", box_raw)
