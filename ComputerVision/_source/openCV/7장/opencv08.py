# 모폴로지 연산 - 침식
import cv2
import numpy as np

img = cv2.imread("./_threshod_foxes.jpg")
ker = np.ones((5, 5), np.uint8)
new_img = cv2.erode(img, ker, iterations=1)
cv2.imshow("erode", new_img)
cv2.waitKey()
