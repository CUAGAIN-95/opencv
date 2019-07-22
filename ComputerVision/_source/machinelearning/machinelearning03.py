# 서포트 벡터 머신 SVM

from sklearn import datasets, metrics, svm
from skimage import io
from numpy import ndarray

mnist = datasets.load_digits()
images = mnist.images
data_size = len(images)

# 이미지 전처리
images = images.reshape(len(images), -1)
labels = mnist.target

# SVM 초기화
SVM_classifier = svm.SVC(gamma=0.001)

# 데이터 세트의 75%의 데이터만으로 학습. 나머지 25%는 SVM을 테스트하는 데 사용됨
SVM_classifier.fit(images[:int((data_size / 4) * 3)], labels[:int((data_size / 4) * 3)])

# 데이터 테스트
predictions = SVM_classifier.predict(images[int((data_size / 4)):])
target = labels[int((data_size / 4)):]

# 학습된 SVM의 성능 출력
print("Performance Report: \n %s \n" % (metrics.classification_report(target, predictions)))
