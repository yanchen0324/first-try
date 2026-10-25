import cv2
import numpy as np

# 用去噪后的图做输入，素描才不会有满屏颗粒
img = cv2.imread("outputs/6_混合滤波.png")
if img is None:
    print("找不到 outputs/6_混合滤波.png，先运行 04_混合滤波.py")
    exit()

gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# 1) 双边平滑：压掉残余噪点，保住边缘
smooth = cv2.bilateralFilter(gray, 9, 90, 90)

# 2) CLAHE：提升局部对比，让暗脸也有层次
clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
eq = clahe.apply(smooth)

# 3) 颜色减淡（Color Dodge）
inv = 255 - eq
blur = cv2.GaussianBlur(inv, (21, 21), 0)
strokes = cv2.divide(eq, 255 - blur, scale=256)

# 4) 乘上灰度色调：暗部变成铅笔灰
tone = (255 - (255 - gray.astype(np.float32)) * 0.28).astype(np.uint8)
sketch = cv2.multiply(strokes, tone, scale=1.0 / 255)

# 保存单张
cv2.imwrite("outputs/9_铅笔素描.png", sketch)

# 拼“原图 | 素描”对比图：灰度要转成 3 通道才能和彩色图左右拼
sketch_bgr = cv2.cvtColor(sketch, cv2.COLOR_GRAY2BGR)
compare = np.hstack([img, sketch_bgr])
cv2.imwrite("outputs/9_对比.png", compare)

print("完成：outputs/9_素描对比.png")
