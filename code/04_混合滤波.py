import cv2
import numpy as np

img = cv2.imread("图1.png")
if img is None:
    print("读不到 图1.png，检查以下当前目录")
    exit()

# 第 1 步：中值 3×3，先去除孤立的噪点
med = cv2.medianBlur(img,3)

# 第 2 步：双边滤波，平滑但保住边缘
bil = cv2.bilateralFilter(med,7,30,10)

# 第 3 步：非局部均值，把剩下的颗粒噪声压到最低
den = cv2.fastNlMeansDenoisingColored(bil, None, 3, 6, 7, 25)


# 保存单张结果
cv2.imwrite("outputs/6_混合滤波.png",den)

# 拼一张“原图 | 去噪后”的对比图
compare = np.hstack([img,den])
cv2.imwrite("outputs/6_混合滤波对比.png",compare)

# 第 4 步：轻度锐化，把被磨软的边缘找回来一点
blur = cv2.GaussianBlur(den,(0,0),1.0)
den = cv2.addWeighted(den,1.5,blur,-0.5,0)
print("完成！")
