import cv2

img = cv2.imread("../../../_image/_foxes.jpg")
cv2.imshow("main", img)
gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
thresh_img = cv2.threshold(gray, 127, 255, 0)
contours, hierarchy = cv2.findContours(thresh_img[1], cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(gray, contours, -1, (255, 0, 0), 1)
cv2.imshow("contours", gray)
cv2.waitKey()
