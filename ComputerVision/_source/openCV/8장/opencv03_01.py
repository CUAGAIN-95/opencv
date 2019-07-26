# 키포인트를 찾은 후 다른 이미지에 대응

import cv2
import random

image = cv2.imread("../../../_image/_foxes.jpg")
image_rot = cv2.imread("../../../_image/_foxes_rot.jpg")

gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
gray_rot = cv2.cvtColor(image_rot, cv2.COLOR_BGR2GRAY)

orb = cv2.ORB_create()
kp, des = orb.detectAndCompute(gray, None)
kp_rot, des_rot = orb.detectAndCompute(gray_rot, None)

# 기본 파라미터로 BFMatcher 생성
bf = cv2.BFMatcher()
matches = bf.knnMatch(des, des_rot, k=2)

# 비율 테스트
good = []
for m, n in matches:
    if m.distance < 0.4 * n.distance:
        good.append([m])

# 대응된 키포인트를 섞음
random.shuffle(good)

# cv2.drawMatchesKnn은 대응 쌍의 리스트들의 리스트를 받는다


for i in range(0, len(good) - 1, 10):
    image_match = cv2.drawMatchesKnn(image, kp, image_rot, kp_rot, good[:i], flags=2, outImg=None)
    print(i)
    cv2.imshow("sift_matches", image_match)
    cv2.waitKey(1000)

cv2.destroyAllWindows()
