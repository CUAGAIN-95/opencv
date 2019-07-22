from skimage import segmentation, color
from skimage import io
from skimage.future import graph
from matplotlib import pyplot as plt

img = io.imread("../../_image/_foxes.jpg")
img_segments = segmentation.slic(img, compactness=20, n_segments=500)
superpixels = color.label2rgb(img_segments, img, kind='avg')
io.imshow(superpixels)
io.show()
