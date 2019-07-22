# 가우시안 블러
import cv2

img = cv2.imread("../../../_image/_foxes.jpg")
new_img = cv2.GaussianBlur(img, (5, 5), 0)
cv2.imshow("GaussianBlur", new_img)
cv2.waitKey()
