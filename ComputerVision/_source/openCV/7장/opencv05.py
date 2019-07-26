# 이미지 회전

import cv2

img = cv2.imread("../../../_image/_foxes.jpg")
r, c ,cannel= img.shape
print(r,c,cannel)

M = cv2.getRotationMatrix2D((233,233), 90, 1)
new_img = cv2.warpAffine(img, M, (r, c))

r, c = new_img.shape[:2]
print(r,c)

cv2.imwrite("rotate_img.jpg", new_img)
cv2.imshow("main", img)
cv2.imshow("turn!", new_img)
cv2.waitKey()
cv2.destroyAllWindows()
