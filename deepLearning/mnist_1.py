import sys, os
sys.path.append(os.pardir)
sys.path.append("C:\\Users\\User\\Desktop\\deepLearning\\7week")
from mnist import load_mnist
from PIL import Image
import pickle
import matplotlib.pylab as plt
from softmaxFunction import softmax

def sigmoid(x):
    return 1 / (1 + np.exp(-x))    

X = np.arange(-5.0, 5.0, 0.1)
Y = sigmoid(X)
plt.plot(X, Y)
plt.ylim(-0.1, 1.1)
plt.show()



(x_train, t_train), (x_test, t_test) = \
    load_mnist(flatten=True, normalize=False)
    
print(x_train.shape)
print(t_train.shape)
print(x_test.shape)
print(t_test.shape)

def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()
    
img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28, 28)
print(img.shape)

img_show(img)

# 데이터를 가져오는 부분
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True, one_hot_label=False)
    return x_test, t_test

# 누군가가 학습시켜놓은 가중치를 가져오는 부분 => 추론
def init_network():
    with open("C:\\Users\\User\\Desktop\\deepLearning\\7week\\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

# 입력층이 1개, 은닉층 1개, 출력층은 softmax 쓰는 predict 함수
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1) # 1층
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2) # 2층
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)

    return y


x, t = get_data() # x, t 변수에 데이터를 가져옴
network = init_network()

# 예측 성능평가
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy:" + str(float(accuracy_cnt) / len(x)))