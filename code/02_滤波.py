import cv2

#读图
img = cv2.imread("图1.png")

if img is None:
    print("读不到 图1.png，确认一下是否在 图像处理作业 文件夹中运行")
    exit()
    
print("原图尺寸：",img.shape)

#高斯滤波   5×5 窗口，sigmaX=0 表示标准差让 OpenCV 自动算
gauss = cv2.GaussianBlur(img,(5,5),0)

#中值滤波     窗口 5×5（必须是奇数），取中位数
median = cv2.medianBlur(img,5)

#方框滤波     -1表示输出类型和原图一样；normalize=True表示除以25
box = cv2.boxFilter(img,-1,(5,5),normalize=True)

#均值滤波     归一化后的方框滤波
mean = cv2.blur(img,(5,5))

cv2.imwrite("outputs/1_高斯滤波.png",gauss)
cv2.imwrite("outputs/2_中值滤波.png",median)
cv2.imwrite("outputs/3_方框滤波.png",box)
cv2.imwrite("outputs/4_均值滤波.png",mean)

print("四种滤波完成，图片保存在/outputs里")
