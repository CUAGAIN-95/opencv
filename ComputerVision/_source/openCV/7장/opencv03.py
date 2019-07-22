# 이미지 크기조정

import cv2
img = cv2.imread("../../../_image/_foxes.jpg")
r,c = img.shape[:2]
new_img = cv2.resize(img,(2*r,2*c),interpolation=cv2.INTER_CUBIC)
cv2.imwrite("new_foxes.jpg",new_img)
cv2.imshow("new_big",new_img)
cv2.waitKey()