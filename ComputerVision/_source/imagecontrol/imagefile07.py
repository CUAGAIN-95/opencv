from skimage import measure
from skimage.io import imread
from skimage.color import rgb2gray
from skimage.filters import sobel

import matplotlib.pyplot as plt

# 이미지읽기
img = imread("../../_image/_foxes.jpg")
img_gray = rgb2gray(img)
img_edge = sobel(img_gray)

# 이미지 윤곽선 검출
contours = measure.find_contours(img_edge, 0.2)

# 이미지와 찾은 윤곽선 표시
fig, ax = plt.subplots()
ax.imshow(img_edge, interpolation='nearest', cmap=plt.cm.gray)
for n, contour in enumerate(contours):
    ax.plot(contour[:, 1], contour[:, 0], linewidth=2)
ax.axis('image')
ax.set_xticks([])
ax.set_yticks([])
plt.show()
