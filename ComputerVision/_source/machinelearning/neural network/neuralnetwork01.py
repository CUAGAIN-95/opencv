from sklearn.datasets import fetch_openml
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import normalize
from sklearn.model_selection import train_test_split

# MNIST 데이터를 가져옴
print('Getting MNIST data....')
mnist = fetch_openml('mnist_784', version=1)
print('MNIST Data downloaded!')
images = mnist.data
labels = mnist.target

# 이미지 전처리
images = normalize(images, norm='l2')  # l1 norm도 사용가능

# 데이터를 학습 세트와 테스트 세트로 나눔
images_train, images_test, labels_train, labels_test = train_test_split(images, labels, test_size=0.25,
                                                                        random_state=17)
# 학습하기 위한 신경망 설정
nn = MLPClassifier(hidden_layer_sizes=(100), max_iter=30, solver='sgd', learning_rate_init=0.001, verbose=True)

# 네트워크 학습 시작
print('NN Training started...')
nn.fit(images_train, labels_train)
print('NN Training completed!')

# 테스트 데이터로 신경망 성능 평가
print('Network Performance %f' % nn.score(images_test, labels_test))
