# 원본 이미지와 변형 이미지 대응

from skimage import data
from skimage import transform as tf
from skimage.feature import (match_descriptors, corner_harris, corner_peaks, ORB, plot_matches)
from skimage.color import rgb2gray

import matplotlib.pyplot as plt

# 원본 이미지 읽기
image_org = data.astronaut()
# 그레이스케일로 변환
image_org = rgb2gray(image_org)
# 이미지를 회전시켜서 이미지를 준비함. 특징점 대응을 보여주기 위함
image_rot = tf.rotate(image_org, 180)
# 이미지에 아핀 변환을 적용해 다른 이미지 생성
tform = tf.AffineTransform(scale=(1.3, 1.1), rotation=0.5, translation=(0, -200))
image_aff = tf.warp(image_org, tform)
# ORB 특징점 기술자를 초기화
descriptor_extractor = ORB(n_keypoints=200)
# 원본 이미지에서 특징 추출
descriptor_extractor.detect_and_extract(image_org)
keypoints_org = descriptor_extractor.keypoints
descriptors_org = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(image_rot)
keypoints_rot = descriptor_extractor.keypoints
descriptors_rot = descriptor_extractor.descriptors

descriptor_extractor.detect_and_extract(image_aff)
keypoints_aff = descriptor_extractor.keypoints
descriptors_aff = descriptor_extractor.descriptors

matches_org_rot = match_descriptors(descriptors_org, descriptors_rot, cross_check=True)
matches_org_aff = match_descriptors(descriptors_org, descriptors_aff, cross_check=True)

fig, ax = plt.subplots(nrows=2, ncols=1)
plt.gray()

plot_matches(ax[0], image_org, image_rot, keypoints_org, keypoints_rot, matches_org_rot)
ax[0].axis('off')
ax[0].set_title("Original Image vs. Transformed Image")

plot_matches(ax[1], image_org, image_aff, keypoints_org, keypoints_aff, matches_org_aff)
ax[1].axis('off')
ax[1].set_title("Original Image vs. Transformed Image")

plt.show()
