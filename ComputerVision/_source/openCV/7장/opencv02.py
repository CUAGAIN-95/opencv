# 이미지 컬러변경

import cv2

img = cv2.imread("../../../_image/_foxes.jpg")
cv2.imshow("main", img)
#컬러 2 그레이, 그레이 2 컬러(불가능)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
cv2.imshow("gray", gray)
gray_return = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
cv2.imshow("gray_return", gray_return)

#컬러 2 HSV, HSV 2 컬러 (가능)
hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
cv2.imshow("hsv", hsv)
hsv_return = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
cv2.imshow("hsv_return", hsv_return)

cv2.waitKey()
