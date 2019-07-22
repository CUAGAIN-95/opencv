# 이미지 회전

import cv2


img = cv2.imread("../../../_image/_foxes.jpg")
c, r = img.shape[:2]
cv2.imshow("main",img)
M = cv2.getRotationMatrix2D((c / 2, r / 2), 90, 1)
new_img = cv2.warpAffine(img, M, (c, r))
cv2.imshow("turn!", new_img)
cv2.waitKey()
