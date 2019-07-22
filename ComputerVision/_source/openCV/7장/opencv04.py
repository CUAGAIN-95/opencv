# 이미지 이동

import cv2
import numpy as np

img = cv2.imread("../../../_image/_foxes.jpg")
r, c = img.shape[:2]
M = np.float32([[1, 0, 100], [0, 1, 100]])
new_img = cv2.warpAffine(img, M, (c, r))
cv2.imshow("move!", new_img)
cv2.waitKey()
