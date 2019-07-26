# 키포인트를 찾은 후 다른 이미지에 대응

import cv2
import random

image = cv2.imread("../../../_image/_foxes.jpg")
image_rot = cv2.rotate(image, rotateCode=90)

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(gray, None)

# sift_obj = cv2.xfeatures2d.SIFT_create()
# keypoints = sift_obj.detect(gray, None)

img = cv2.drawKeypoints(image, kp, None, (255, 0, 0))
cv2.imwrite("../../../_image/_sift_foxes.jpg", img)
cv2.imshow("sift_keypoints.jpg", img)
cv2.waitKey()
cv2.destroyAllWindows()
