from skimage import data, segmentation, color
from skimage.io import imread, imshow, show
from skimage.future import graph

img = imread("../../_image/_foxes.jpg")
img_segments = segmentation.slic(img, compactness=20, n_segments=500)
out1 = color.label2rgb(img_segments, img, kind='avg')
segment_graph = graph.rag_mean_color(img, img_segments, mode='similarity')
img_cuts = graph.cut_normalized(img_segments, segment_graph)
normalized_cut_segments = color.label2rgb(img_cuts, img, kind='avg')

imshow(normalized_cut_segments)
show()
