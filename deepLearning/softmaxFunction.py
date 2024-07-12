import numpy as np

a = np.array([0.3, 2.9, 4.0])

exp_a= np.exp(a)
print(exp_a)

sum_exp_a = np.sum(exp_a) # 지수함수
print(sum_exp_a)

y = exp_a / sum_exp_a
print(y)

def softmax(a):
    c = np.max(a)
    exp_a= np.exp(a-c) # 오버플로 대책
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

'''
결과 해석
0~1 사이 값으로 softmax 함수가 변환을 해주는 것으로,
class y1, y2, y3 ... 중 제일 확률이 높은 것 = 해당 class에 속할 가능성 높음
따라서, 결과가 0.018, 0.25, 0.74로 나왔으므로 y3인 4.0에 속함
'''


a = np.array([0.3, 2.9, 4.0])
y = softmax(a)
print(y)

np.sum(y) # res : 1.0
