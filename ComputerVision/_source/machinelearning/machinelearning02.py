# X

from sklearn import datasets, metrics
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from skimage import io, color, feature, transform

mnist = datasets.load_digits()
images = mnist.images
data_size = len(images)
# 이미지전처리
images = images.reshape(len(images), -1)
labels = mnist.target

# 로지스틱 회기 초기화
LR_classifier = LogisticRegression(C=0.01, penalty='l1', tol=0.01)
# 데이터 세트의 75%만 학습에 사용. 나머지는 로지스틱 회귀 테스트에 이용될 것임
LR_classifier.fit(images[:int((data_size / 4) * 3)], labels[:int((data_size / 4) * 3)])

# 사용자가 제공한 이미지를 불러옴
digit_img = io.imread("../../../_image/_foxes.jpg")
# 이미지를 그레이스케일로 변경
digit_img = color.rgb2gray(digit_img)
# 이미지를 28X28로 크기 조정
digit_img = transform.resize(digit_img, (8, 8), mode="wrap")
# 이미지에 에지검출
digit_edge = feature.canny(digit_img, sigma=5)
digit_edge = [digit_edge.flatten()]

# 데이터 테스트
prediction = LR_classifier.predict(digit_edge)

print(prediction)
