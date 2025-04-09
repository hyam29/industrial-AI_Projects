from torch import nn

class Net(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer = nn.Sequential(
            nn.Linear(1, 2),
            nn.Sigmoid()
        )

        self.fc = nn.Linear(2, 1)
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.constant_(module.bias, 0.01)
        print(f"Apply : {module}")

model = Net()


import torch
from torchvision.datasets import MNIST

from torchvision import transforms
transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize((0.5,), (0.5,))
])


from torchvision.datasets import MNIST

train_data = MNIST(root='data', train=True, download=True, transform=transform)
test_data = MNIST(root='data', train=False, download=True, transform=transform)

print(len(train_data), len(test_data))  # (60000, 10000)


from torch.utils.data import DataLoader, SubsetRandomSampler
import numpy as np

batch_size = 50
valid_size = 0.2

num_train = len(train_data)
indices = list(range(num_train))
np.random.shuffle(indices)

split = int(np.floor(valid_size * num_train))
train_idx, valid_idx = indices[split:], indices[:split]

train_sampler = SubsetRandomSampler(train_idx)
valid_sampler = SubsetRandomSampler(valid_idx)

train_loader = DataLoader(train_data, batch_size=batch_size, sampler=train_sampler)
valid_loader = DataLoader(train_data, batch_size=batch_size, sampler=valid_sampler)
test_loader = DataLoader(test_data, batch_size=batch_size)


for data, target in train_loader:
    print(data.shape)   # torch.Size([50, 1, 28, 28])
    print(target.shape) # torch.Size([50])
    break


from torch import nn
import torch.nn.functional as F

class Model(nn.Module):
    def __init__(self):
        super().__init__()
        self.fc1 = nn.Linear(784, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, 128)
        self.fc4 = nn.Linear(128, 56)
        self.fc5 = nn.Linear(56, 10)

    def forward(self, x):
        x = x.view(x.shape[0], -1)  # Flatten
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        return x

model = Model()


import torch
from torch import optim

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.01)


for epoch in range(1, 11):
    train_loss, valid_loss = [], []

    # Training
    model.train()
    for data, target in train_loader:
        optimizer.zero_grad()
        output = model(data)
        loss = criterion(output, target)
        loss.backward()
        optimizer.step()
        train_loss.append(loss.item())

    # Validation
    model.eval()
    with torch.no_grad():
        for data, target in valid_loader:
            output = model(data)
            loss = criterion(output, target)
            valid_loss.append(loss.item())

    print(f"Epoch: {epoch}, Training Loss: {np.mean(train_loss)}, Valid Loss: {np.mean(valid_loss)}")


nn.init.normal_(self.fc1.weight, mean=0, std=1)  # 정규분포 초기화
nn.init.kaiming_normal_(self.fc1.weight)         # He 초기화
# Lecun은 PyTorch 기본값 (별도 설정 필요 없음)


test_loss = 0.0
class_correct = list(0. for _ in range(10))
class_total = list(0. for _ in range(10))

model.eval()
for data, target in test_loader:
    output = model(data)
    loss = criterion(output, target)
    test_loss += loss.item() * data.size(0)

    _, pred = torch.max(output, 1)
    correct = np.squeeze(pred.eq(target.data.view_as(pred)))

    for i in range(batch_size):
        label = target.data[i]
        class_correct[label] += correct[i].item()
        class_total[label] += 1

test_loss /= len(test_loader.dataset)
print('Test Loss: {:.6f}'.format(test_loss))

for i in range(10):
    if class_total[i] > 0:
        print(f'Test Accuracy of {i}: {100 * class_correct[i] / class_total[i]:.0f}%')
print(f'\nTest Accuracy (Overall): {100. * np.sum(class_correct) / np.sum(class_total):.0f}%')


!pip install numpy requests nlpaug transformers sacremoses nltk

import nlpaug.augmenter.word as naw
import nltk

texts = [
    'Those who can imagine anything, can create the impossible.',
    'We can only see a short distance ahead, but we can see plenty there that to be done.',
    'If a machine is expected to be infallible, It cannot also be intelligent.',
]

aug = naw.ContextualWordEmbsAug(model_path='bert-base-uncased', action='insert')
augmented_texts = aug.augment(texts)

for text, augmented in zip(texts, augmented_texts):
  print(f'src : {text}')
  print(f'dst : {augmented}')
  print('-------------------------------')

!pip install imgaug

!pip install numpy==1.24.4 # colab imgaug 라이브러리 버전 문제로 numpy 다운그레이드

import numpy as np
np.bool = np.bool_ # Deprecated 오류 방지
from PIL import Image
from torchvision import transforms
from imgaug import augmenters as iaa

import matplotlib.pyplot as plt


class IaaTransforms:
  def __init__(self):
    self.seq = iaa.Sequential([
        iaa.SaltAndPepper(p=(0.03, 0.07)),
        iaa.Rain(speed=(0.3, 0.7))
    ])

  def __call__(self, images):
    images = np.array(images)
    augmented = self.seq.augment_image(images)
    return Image.fromarray(augmented)


transform = transforms.Compose([
    IaaTransforms()
])


image = Image.open('/content/drive/MyDrive/ex_images/11.PNG')
transformed_image = transform(image)
plt.imshow(transformed_image)


