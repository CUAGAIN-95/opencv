import cv2

img = cv2.imread("../../../_image/_foxes.jpg")
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
x_edges = cv2.Sobel(gray, -1, 1, 0, ksize=5)
y_edges = cv2.Sobel(gray, -1, 0, 1, ksize=5)
cv2.imshow("x", x_edges)
cv2.imshow("y", y_edges)
cv2.waitKey()
