from skimage.feature import ORB, match_descriptors
from skimage.io import imread
from skimage import io
from skimage.measure import ransac
from skimage.transform import ProjectiveTransform, warp, SimilarityTransform
from skimage.color import rgb2gray, gray2rgb
from skimage.exposure import rescale_intensity
import numpy as np

image0 = imread("../../_image/house1.png")
image0 = rgb2gray(image0)
image1 = imread("../../_image/house2.png")
image1 = rgb2gray(image1)

orb = ORB(n_keypoints=1000, fast_threshold=0.05)

orb.detect_and_extract(image0)
keypoints1 = orb.keypoints
descriptors1 = orb.descriptors

orb.detect_and_extract(image1)
keypoints2 = orb.keypoints
descriptors2 = orb.descriptors

matches12 = match_descriptors(descriptors1, descriptors2, cross_check=True)
src = keypoints2[matches12[:, 1]][:, ::-1]
dst = keypoints1[matches12[:, 0]][:, ::-1]
transform_model, inliers = ransac((src, dst), ProjectiveTransform, min_samples=4, residual_threshold=2)
r, c = image1.shape[:2]

corner = np.array([[0, 0], [0, r], [c, 0], [c, r]])
warped_corners = transform_model(corner)
all_corners = np.vstack((warped_corners, corner))
corner_min = np.min(all_corners, axis=0)
corner_max = np.max(all_corners, axis=0)

output_shape = (corner_max - corner_min)
output_shape = np.ceil(output_shape[::-1])

offset = SimilarityTransform(translation=-corner_min)

image0_warp = warp(image0, offset.inverse, output_shape=output_shape, cval=-1)
image1_warp = warp(image1, (transform_model + offset).inverse, output_shape=output_shape, cval=-1)

image0_mask = (image0_warp != -1)
image0_warp[~image0_mask] = 0
image0_alpha = np.dstack((gray2rgb(image0_warp), image0_mask))

image1_mask = (image1_warp != -1)
image1_warp[~image1_mask] = 0
image1_alpha = np.dstack((gray2rgb(image1_warp), image1_mask))

merged = (image0_alpha + image1_alpha)
alpha = merged[..., 3]
merged /= np.maximum(alpha, 1)[..., np.newaxis]
io.imshow(merged)
io.show()
io.imsave('output.png', merged)
