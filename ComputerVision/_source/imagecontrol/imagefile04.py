from skimage.transform import rotate
from skimage.feature import local_binary_pattern
from skimage import data
import numpy as np

# 알고리즘을 테스트하기 위해 세 개의 다른 이미지를 불러옴
brick = data.load('brick.png')
grass = data.load('grass.png')
wall1 = data.load('rough-wall.png')

# 세 개의 이미지에 대해 LBP 특징 계산
brick_lbp = local_binary_pattern(brick, 16, 2, 'uniform')
grass_lbp = local_binary_pattern(grass, 16, 2, 'uniform')
wall1_lbp = local_binary_pattern(wall1, 16, 2, 'uniform')

# 이 이미지들을 22도 회전
brick_rot = rotate(brick, angle=22, resize=False)
grass_rot = rotate(grass, angle=22, resize=False)
wall1_rot = rotate(wall1, angle=22, resize=False)

# 회전된 모든 이미지에서 LBP 특징을 계산
brick_rot_lbp = local_binary_pattern(brick_rot, 16, 2, 'uniform')
grass_rot_lbp = local_binary_pattern(grass_rot, 16, 2, 'uniform')
wall1_rot_lbp = local_binary_pattern(wall1_rot, 16, 2, 'uniform')

# 벽돌 이미지를 선택하고 회전된 이미지들 중에서 가장 잘 대응되는 이미지를 찾음
# LBP 특징을 담은 목록 생성
bins_num = int(brick_lbp.max() + 1)
brick_hist, _ = np.histogram(brick_lbp, bins=bins_num, range=(0, bins_num))
lbp_features = [brick_rot_lbp, grass_rot_lbp, wall1_rot_lbp]

min_score = 1000  # Set a very large best score value initially
winner = 0  # 가장 잘 대응되는 이미지의 인덱스를 저장하기 위한 값
idx = 0

for feature in lbp_features:
    histogram, _ = np.histogram(feature, bins=bins_num, range=(0, bins_num))
    p = np.asarray(brick_hist)
    q = np.asarray(histogram)
    filter_idx = np.logical_and(p != 0, q != 0)
    score = np.sum(p[filter_idx] * np.log2(p[filter_idx] / q[filter_idx]))
    if score < min_score:
        min_score = score
        winner = idx
    idx = idx + 1

if winner == 0:
    print('\n' + 'Brick matched with Brick Rotated')
elif winner == 1:
    print('\n' + 'Brick matched with Grass Rotated')
else:
    print('\n' + 'Brick matched with Wall Rotated')
