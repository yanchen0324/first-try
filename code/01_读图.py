import cv2

#1.读图片
img = cv2.imread("图1.png")

#2.看一下读到了什么
print("图片形状：",img.shape)
print("宽度  x 高度：",img.shape[1],"x",img.shape[0])

#3.显示图片
cv2.imshow("图1",img)
cv2.waitKey(0)
cv2.destroyAllWindows()
