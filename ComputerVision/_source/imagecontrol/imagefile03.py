# from PIL import Image
# from PIL import ImageFilter
#
# img = Image.open("D:\_lyh\_Image/_cat.jpg")
# blur_img = img.filter(ImageFilter.GaussianBlur(5))
# blur_img.show()

from skimage import io, filters

img = io.imread("../../_image/_foxes.jpg")
out = filters.gaussian(img, sigma=5)
io.imshow(out)
io.show()
