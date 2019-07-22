from sklearn import datasets, metrics
from sklearn.linear_model import LogisticRegression

mnist = datasets.load_digits()
images = mnist.images
data_size = len(images)

# 이미지 전처리
images = images.reshape(len(images), -1)
labels = mnist.target

# 로지스틱 회귀 초기화
LR_classifier = LogisticRegression(C=0.01, penalty='l1', tol=0.01)

# 데이터 세트의 75%만 학습에 사용. 나머지는 로지스틱 회귀 테스트에 이용될 것임
LR_classifier.fit(images[:int((data_size / 4) * 3)], labels[:int((data_size / 4) * 3)])

# 데이터 테스트
predictions = LR_classifier.predict(images[int((data_size / 4)):])
target = labels[int((data_size / 4)):]

# 학습된 로지스틱 회귀 모델의 성능 출력
print("Performance Report: \n %s \n" % (metrics.classification_report(target, predictions)))
