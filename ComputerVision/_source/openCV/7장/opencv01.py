# 이미지 읽기

import cv2

img = cv2.imread("../../../_image/_foxes.jpg")
cv2.imshow("HI", img)
img_crop = img[200:400, 150:350]
cv2.imwrite("template.jpg", img_crop)
cv2.waitKey()
