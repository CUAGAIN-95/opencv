# 케니 에지검출

import cv2

img = cv2.imread("../../../_image/_foxes.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
edges = cv2.Canny(gray, 100, 200)
cv2.imshow("Canny", edges)
edges1 = cv2.Canny(gray, 100, 150, 3)
cv2.imshow("Canny1", edges1)
edges2 = cv2.Canny(gray, 100, 200, 5)
cv2.imshow("Canny2", edges2)
cv2.waitKey()
