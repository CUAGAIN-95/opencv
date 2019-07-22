# 모폴로지 연산 - 팽창
import cv2
import numpy as np

img = cv2.imread("./_threshod_foxes.jpg")
ker = np.ones((5, 5), np.uint8)
new_img = cv2.dilate(img, ker, iterations=1)
cv2.imshow("dilation", new_img)
cv2.waitKey()
