from sklearn import datasets, metrics
from sklearn.cluster import KMeans

mnist = datasets.load_digits()
images = mnist.images
data_size = len(images)

# 이미지 전처리
images = images.reshape(len(images), -1)
labels = mnist.target

# Kmeans 초기화
clustering = KMeans(n_clusters=10, init='k-means++', n_init=10)
# 전체 데이터의 75%를 사용해 학습. 나머지 25%는 k-평균 클러스터링 테스트에 이용
clustering.fit(images[:int((data_size / 4) * 3)])
# 클러스터들의 중심점 출력
print(clustering.labels_)
# 데이터 테스트
predictions = clustering.predict(images[int((data_size / 4) * 3):])
