---
title: MobileNetV2
categories:
   - CV
tags:
   - CV
---

# MobileNet v2

[Reference]

논문 링크: **[MobileNets: Inverted Residuals and Linear Bottlenecks](https://arxiv.org/abs/1801.04381)**

Github: https://github.com/tensorflow/models/blob/master/research/slim/nets/mobilenet/README.md

- 2018년 1월(Arxiv)
- Google Inc.
- Mark Sandler, Andrew Howard, Menglong Zhu, Andrey Zhmoginov, Liang-Chieh Chen

Blog: https://greeksharifa.github.io/computer%20vision/2022/02/10/MobileNetV2/

---

## 요약

- Inverted Residual 구조를 이용해 구성된 Bottleneck layer 사용
- 정보손실을 줄이기 위해 Narrow layer에서는 비선형 Activation Function을 제거
- 성능대비 파라미터가 적어, 메모리 사용이 적기때문에 임베디드 보드에서 사용하기 좋음
- SSDLite에 붙여서 Object detection, DeepLabv3에 붙여서 Semantic segmentation 가능

---

## 용어 정리

- ```ImageNet```: ```Image Network 구조가 아닌, ImageNet Dataset을 의미함```
- ```Spatial```: ```공간``` ```x축, y축과 같은 공간```
- ```Embedding```: ```고차원의 정보를 저차원에 매핑(또는 저차원으로 표현)```
- ```Manifold```: ```고차원의 정보를 잘 아우르는 저차원(Subspace)``` ``` 원래 정보를 잘 유지하며 차원축소``` [내 게시글로 링크 만들기]()
- ```Activation Function``` : ``` ReLu와 같은 비선형 함수``` [내 게시글로 링크 하나 만들기]()
- ```Skip Connection```: ```Residual Connection``` ```Gradient Vanishing Problem 완화```
- ```Depthwise Separable Convolution``` = ```Depthwise Convolution + Pointwise Convolution```
  - ```Depthwise Convolution ``` : ```채널별 Convolution 연산```
  - ```Pointwise Convolution``` : ``` 1x1 Convolution = Projection Convolution 연산```

---

## 설명

### 1. Convolution

> MobileNetV2에선 Standard Convolution을 대체할 수 있고, 연산량은 더 적은 Depthwise Separable Convolution 연산을 사용

#### 1-1. Standard(Conventional) Convolution

![image-20221127165223954](https://raw.githubusercontent.com/ech97/save-image-repo/master/image/image-20221127165223954.png)

> **Input**의 각 Data(①, ②, ③, ④)와 **Kernel1**의 각 Data(⑤, ⑥, ⑦, ⑧)간의 원소별 곱셈(①\*⑤, ②\*⑥, ③\*⑦, ④\*⑧)의 **합**이 곧 **Output**의 x1 (Padding 생략)

따라서 $k \cdot k \cdot w_i \cdot h_i \cdot d_i \cdot d_j$ 번의 연산 발생

#### 1-2. Depthwise Convolution

![image-20221127165234000](https://raw.githubusercontent.com/ech97/save-image-repo/master/image/image-20221127165234000.png)

> 각 채널별로 Convolution 연산 진행 (Padding 생략)

**따라서 $k \cdot k \cdot w_i \cdot h_i \cdot d_i$ 번의 연산 발생**

#### 1-3. Pointwise Convolution

![image-20221127165336948](https://raw.githubusercontent.com/ech97/save-image-repo/master/image/image-20221127165336948.png)

> 채널을 임의의 수 $d_j$로 조정가능 (Padding 생략)

$w_i \cdot h_i \cdot d_i \cdot d_j$ 번의 연산 발생

#### 1-4. Depthwise Saparable Convolution

> Depthwise Convolution연산뒤에 Pointwise Convolution 연산을 Saparable하게 진행하므로 연산의 수는  $k \cdot k \cdot w_i \cdot h_i \cdot d_i$ $\cdot$ $w_i \cdot h_i \cdot d_i \cdot d_j$ 

따라서 $h_i \cdot w_i \cdot d_i (k^2+d_j)$ 번의 연산이 발생하며 Standard Convolution에 대해 ${1 \over d_j}+{1 \over k^2}$ 만큼의 연산량 감소

---

### 2. Linear Bottlenecks

> 데이터의 정보를 **Embedding(압축)** 하는 **과정(Bottleneck)**에서 정보의 손실을 최소화하기 위해 Non-Linear 연산 대신 **Linear연산** 진행

![img](https://raw.githubusercontent.com/ech97/save-image-repo/master/image/img.png)

- [1-1. Standard Convolution](#1-1-standard-convolution) 에서 알 수 있듯이, 학습된 몇몇 Feature map에서만 **의미있는 Manifold(벡터 매핑)**를 획득함

  - 의미있는 Manifold를 다시 Embedding시켜서 **Manifold of Interest(우리가 관심을 가지고있는 Manifold) 부분**이 **Entire space**가 될때까지 차원 축소 가능

- 하지만 이 과정에서 사용하는 **ReLU**와 같은 **Non-Linear** 함수에 의해 데이터 유실 발생 (0이하의 값은 제거하기 때문에)

  - 이를 방지하고자 **Channel expansion**을 통해, 어떤 채널에선 없어진 정보들이 다른 채널에서는 살아있도록하여 정보의 손실가능성을 최소화함

- 따라서 [1-3. Pointwise Convolution](#1-3-pointwise-convolution)의 채널 압축/팽창 과정에서

  - $d_i \le d_j$ (채널 팽창) 일 땐, **Non-Linear** 함수를 사용해도 정보 손실 없음

  - $d_i > d_j$ (채널 압축) 일 땐, **Non-Linear** 함수를 사용하게 되면, 정보 손실이 발생하므로 **Linear**함수를 사용해야함

    - 고차원의 그래프를 2D space에 Projection(사영)한 이미지

      > ReLU를 이용하여 dim을 압축하는 경우, 다음과 같이 많이 압축할수록 정보 손실이 많이 발생

    ![img](https://blog.kakaocdn.net/dn/bdzecw/btqW1ssLQmk/xwlCyuRwWfSaucZKr3emkK/img.png)

    -  Pointwise Convolution(Projection Convolution)연산에 Non-linear 함수의 유/무에 따른 성능 비교

    <img src="https://blog.kakaocdn.net/dn/bu0fvo/btqW1tZvOVG/uY67qm9PonsuIsG8qe8Ugk/img.png" alt="img"   style="zoom:80%;" />

---

### 3. Inverted Residual

> 기존 Residual은 wide1 - narrow - wide2 layer와, wide2에 wide1를 더해주는 방식인데,
>
> Inverted Residual 방식은 narrow1 - wide - narrow2 layer와, narrow2에 narrow1을 더해주는 방식

![image-20220708104322265](https://raw.githubusercontent.com/ech97/save-image-repo/master/image/image-20220708104322265.png)

- narrow1에 이미 필요한 정보들이 압축되어있다고 가정하고있기 때문에, Skip connection (Residual 하게) narrow2에 narrow1을 더 함

  - Skip Connection을 통해 Gradient vanishing 문제 완화
- Depthwise Convolution 연산을 진행할 때에는 ReLU함수를 사용하지만, Pointwise Convolution 연산을 진행할 때는 Non-Linear함수를 사용하지 않음
- 기존 Residual 방식에 비해 Inverted Residual 방식의 연산량이 더 적음

  - Input Channel $K$를 Expansion Ratio를 사용하여, $t \cdot K$로 확장시켰을때 필요한 계산량 $= h \cdot w \cdot d_i \cdot t(d_i + k^2 + d_j)$


---

### 4. Model Architecture

- 기본 구조: *Bottleneck residual block*

<img src="https://raw.githubusercontent.com/ech97/save-image-repo/master/image/tab01.png" alt="img"   style="zoom:50%;" />

- 19개의 *Bottleneck residual block* 과 1개의 *Fully Convolution Layer*로 이루어져 있다
  - *t*: Expansion ratio/채널확장계수 ``` 논문에서 Inverted residual의 Expasion ratio는 6으로 고정```
  - *c*: Channel
  - *n*: Iteration
  - *s*: Stride ``` 각 Sequence마다 첫번째 Stride는 s이며, 나머지 경우에 대해 Stride는 1```

<img src="https://raw.githubusercontent.com/ech97/save-image-repo/master/image/image-20220708105912317.png"   alt="image-20220708105912317" style="zoom:80%;" />

- Trade-off hyper parameter

  - Multiplier 1 ($224 \times 224$) 일 때
    - $300M$ Multiply-Adds (*MAdds*)
    - $3.4M$ Paramter
  - Multiplier 0.35 ~ 1.4 (Resolution 96 ~ 224)
    - 계수별 연산량과 정확도 Top 1

  <img src="https://raw.githubusercontent.com/ech97/save-image-repo/master/image/image-20220708114636611.png" alt="image-20220708114636611"   style="zoom:80%;" />

---

### 5. Memory efficient inference

> 사용되는 메모리의 총량은 Bottleneck 내부의 Tensor보다는 Bottleneck tensor의 크기에 지배된다
>
> 또한 Cache miss를 줄이기 위해, Expansion ratio를 2에서 5사이로 조정하는 것이 좋으나, Framwork의 Optimization 성능에 따라 달라질 수 있음

---

### 6. Experiments

#### 6-1. ImageNet Classification

- RMSPropOptimizer, Tensorflow, weight decay 0.00004, learning rate 0.045, 16 GPU, batch size 96
- 마지막 열의 running time은 TF-Lite 모델로 변환시켜 Google Pixel1에서 실행시킨 결과

<img src="https://raw.githubusercontent.com/ech97/save-image-repo/master/image/image-20220708114838852.png"   alt="image-20220708114838852" style="zoom:80%;" />

#### 6-2. COCO Object Detection

- SSDLite

  - 기존 SSD의 Convolution 연산을 모두 Depthwise Separable Convolution 연산으로 변경

    - 엄청난 양의 Parameter 감소

      <img src="https://raw.githubusercontent.com/ech97/save-image-repo/master/image/image-20220708115522401.png" alt="image-20220708115522401"   style="zoom:100%;" />

  - MobileNetV2와 SSDLite를 연결하여 *trainval35k Dataset* 을 이용하여 학습

    - Parameter 감소 및 매우 낮은 실행시간

      > TFLite를 사용하여 Google Pixel1에서 실행

      <img src="https://raw.githubusercontent.com/ech97/save-image-repo/master/image/image-20220708115930720.png" alt="image-20220708115930720"   style="zoom:100%;" />

#### 6-3. Semantic Segmentation

- MobileNetV2 + DeepLabv3

  - DeepLabv3는 Atours Convolution 연산을 진행하여, Resolution이 좋아 Semantic Segmentation에 유리

  - MobileNetV2를 Feature Extractor로 사용

    - 모델은 *COCO*로 Pretrain 되었으며, *PASCAL VOC 2012*를 이용하여 평가함

      > ResNet base 모델에 비해 우수한 성능과 적은 Parameter를 보임

      <img src="https://raw.githubusercontent.com/ech97/save-image-repo/master/image/image-20220708123006020.png" alt="image-20220708123006020"   style="zoom:80%;" />

---

## 구현

> [Code Reference]
>
> Blog: https://visionhong.tistory.com/17

### 1. 코드 개요

> - [Kaggle의 Intel Image](https://www.kaggle.com/datasets/puneet6060/intel-image-classification) 분류 모델 제작
> - 6 종류의 장면 구분 (Street, sea, mountain, forest, glacier, buildings)

![image-20220725103334666](https://raw.githubusercontent.com/ech97/save-image-repo/master/image/image-20220725103334666.png)

- 기존 MobileNetV2에 ImageNet Data를 넣어 학습을 시켜, 1000개의 데이터를 분류할 수 있게 Pre-Training 시킴
- 1000개를 Classification 할 수 있는 MobileNetV2 Model에 Pre-Train된 Weights를 불러온 뒤
- 마지막 Classification층을 1000개에서 6개로 줄임
- 이때, ImageNet Data는 224x224 사이즈고, Intel Image는 150x150 이므로, Model에 맞게 224x224로 Upscaling 작업 필요
- 총 3가지의 파일

> - mobilenetv2.py: ImageNet Data을 학습시켰던 모델 구성 및 Pre-train된 weights 불러오기
>- mobilenetv2_train.py: Intel Data를 학습시키위해 1000->6 Classification 축소, Image Upscaling 작업 뒤, 학습 진행
> - mobilenetv2_inference.py: Intel Data로 학습된 모델 추론

---

### 2. 코드

> 주석 설명 참고

#### 2-1. mobilenetv2.py

```python
import torch
import torch.nn as nn
import math
import os.path

# 첫번째 Layer에서 사용될 Convolution
def conv_bn(input_channels, output_channels, stride):
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=3,
            stride=stride,
            padding=1,
            bias=False
        ),
        nn.BatchNorm2d(output_channels),
        nn.ReLU6(inplace=True)
    )

# Inverted bottleneck layer 바로 다음에 나오는 Convolution에 쓰일 함수
def conv_1x1_bn(input_channels, output_channels):
    return nn.Sequential(
        nn.Conv2d(
            input_channels,
            output_channels,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=False
        ),
        nn.BatchNorm2d(output_channels),
        nn.ReLU6(inplace=True)
    )

# Channel 수를 항상 8로 나누어 떨어지게 만들어주는 함수
def make_divisible(x, divisible_by=8):
    import numpy as np
    return int(np.ceil(x*1./divisible_by)*divisible_by)


class InvertedResidual(nn.Module):
    def __init__(self, input_channels, output_channels, stride, expand_ratio):
        super(InvertedResidual, self).__init__()
        
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.stride = stride
        assert self.stride in [1, 2]
        self.expand_ratio = expand_ratio
        
        self.hidden_dim = int(self.input_channels * self.expand_ratio) # 증가시킬 Channel 수

        # Skip Connection이 가능한지 확인
        self.use_res_connect = self.stride == 1 and self.input_channels == self.output_channels

        # ★Batch Normalization을 진행할 경우 어차피 채널별로 추가된 편향들은 제거되어서, bias = False 설정
        # 확장하지 않는 경우 == 단순 Depthwise Convolution만 하는 경우
        if self.expand_ratio == 1:
            self.conv = nn.Sequential(
                nn.Conv2d(
                    self.hidden_dim,
                    self.hidden_dim,
                    kernel_size=3,
                    stride=1,
                    padding=1,
                    groups=self.hidden_dim,  # Depthwise Convolution // 채널별 Conv연산
                    bias=False
                    ),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU6(inplace=True), # 제자리에서 연산
                nn.Conv2d(
                        self.hidden_dim,
                        self.output_channels,
                        kernel_size=1,
                        stride=1,
                        padding=0,
                        bias=False
                    ),
                nn.BatchNorm2d(self.output_channels)
            )
        ## 확장하는 경우
        else:
            self.conv = nn.Sequential(
                # 채널 확장
                nn.Conv2d(
                    self.input_channels,
                    self.hidden_dim,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                ),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU6(inplace=True),
                
                # Depthwise Convolution (Kernel size = 3)
                nn.Conv2d(
                    self.hidden_dim,
                    self.hidden_dim,
                    kernel_size=3,
                    stride=self.stride,
                    padding=1,
                    groups=self.hidden_dim,
                    bias=False
                ),
                nn.BatchNorm2d(self.hidden_dim),
                nn.ReLU6(inplace=True),

                # Pointwise Conv를 통해 차원 축소
                nn.Conv2d(
                    self.hidden_dim,
                    self.output_channels,
                    kernel_size=1,
                    stride=1,
                    padding=0,
                    bias=False
                    ),
                nn.BatchNorm2d(self.output_channels)
                # ReLU6는 사용하지 않음 (= Linear)
            )
    
    def forward(self, x):
        if self.use_res_connect:
            return x + self.conv(x)
        else:
            return self.conv(x)


class MobileNetV2(nn.Module):
    def __init__(self, num_classes=1000, input_size=224, width_mult=1.):
        super(MobileNetV2, self).__init__()
        self.block = InvertedResidual    # Class자체임
        self.input_channels = 32
        self.final_channels = 1280
        self.input_size = input_size
        assert self.input_size % 32 == 0

        self.width_mult = width_mult
        self.final_channels = make_divisible(self.input_channels * self.width_mult) if self.width_mult > 1.0 else self.final_channels

        # Feature들을 담을 List에 First Layer 추가        
        self.features = [conv_bn(input_channels=3, output_channels=self.input_channels, stride=2)]
        
        
        self.inverted_residual_setting = [
            # 4번 항목 참고
        
            # t, c, n, s

            # t: expand ratio
            # c: channel
            # n: Number of iterations
            # s = stride
            [1, 16, 1, 1],
            [6, 24, 2, 2],
            [6, 32, 3, 2],
            [6, 64, 4, 2],
            [6, 96, 3, 1],
            [6, 160, 3, 2],
            [6, 320, 1, 1]
        ]

        for t, c, n, s in self.inverted_residual_setting:
            self.output_channels = make_divisible(c * self.width_mult) if t > 1 else c
            for i in range(n):
                if i == 0:  # 첫번째 블록
                    self.features.append(self.block(self.input_channels, self.output_channels, s, t))
                else:   # 첫번째 블록만 Stride 적용하고, 나머지는 1임
                    self.features.append(self.block(self.input_channels, self.output_channels, 1, t))
                self.input_channels = self.output_channels
        
        # 마지막 레이어 제작
        self.features.append(conv_1x1_bn(self.input_channels, self.final_channels)) # 마지막에 채널 뻥튀기 (320 -> 1280)

        # features list를 Sequential로 제작
        self.features = nn.Sequential(*self.features)

        # Average Pooling layer
        self.avg = nn.AvgPool2d(7, 7)
        self.classifier = nn.Linear(self.final_channels, num_classes)
        self._initialize_weights()
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):  # m이 nn.Conv2d의 인스턴스인지 확인
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n)) # 가중치를 평균 0, 편차 sqrt(2/n)으로 초기화
                if m.bias is not None:
                    m.bias.data.zero_()
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                n = m.weight.size(1)
                m.weight.data.normal_(0, 0.01)
                m.bias.data.zero_()
    
    def forward(self, x):
        x = self.features(x)
        x = self.avg(x)
        x = x.view(-1, self.final_channels) # flatten
        x = self.classifier(x)
        return x

def mobilenet_v2(pretrained=True):
    model = MobileNetV2(width_mult=1)

    weights_file = './weights/mobilenetv2_1.pth.tar'
    if pretrained:
        if os.path.isfile(weights_file):
            checkpoint = torch.load(weights_file)
            model.load_state_dict(checkpoint, strict=False)
        else:
            try:
                from torch.hub import load_state_dict_from_url
            except ImportError:
                from torch.utils.model_zoo import load_url as load_state_dict_from_url
            
            state_dict = load_state_dict_from_url(
                'https://www.dropbox.com/s/47tyzpofuuyyv1b/mobilenetv2_1.0-f2a8633.pth.tar?dl=1', progress=True
            )

            model.load_state_dict(state_dict)
    return model

if __name__ == "__main__":
    model = mobilenet_v2()
    from torchsummary import summary
    summary(model, (3, 224, 224))

```

#### 2-2. mobilenetv2_train.py

```python
import os
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import torchvision.datasets as datasets
from mobilenetv2 import mobilenet_v2

device = 'cuda' if torch.cuda.is_available() else 'cpu'

train_path = './dataset/seg_train/seg_train'
test_path = './dataset/seg_test/seg_test'
pred_path = './dataset/seg_pred/seg_pred'

train_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize((.5,.5,.5), (.5,.5,.5))
])

test_transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize((.5,.5,.5), (.5,.5,.5))
])

train_dataset = datasets.ImageFolder(train_path, transform=train_transform)
test_dataset = datasets.ImageFolder(test_path, transform=test_transform)

# pin Memory는 GPU 메모리에 Tensor 복사
# num_workers 는 실행할때 사용할 프로세스 개수
train_loader = DataLoader(train_dataset, batch_size=128, shuffle=True, num_workers=6, pin_memory=False)
test_loader = DataLoader(test_dataset, batch_size=128, shuffle=False, num_workers=6, pin_memory=False)

classes = os.listdir(train_path)    # Class 별로 폴더가 정리되어 있기 때문

model = mobilenet_v2(True)

# model.classifier.in_features = model(MobilenetV2)에서 Classifier의 입력 채널수를 리턴해줌
# 아래 코드를 통해 Classfier 부분을 내가 원하는 Classes 수로 치환
model.classifier = nn.Linear(model.classifier.in_features, len(classes)).to(device)


# 원하는 Device(CPU, GPU)에서 동작시키기위해선,
# 'model'과 'data' 모두 device에 있어야한다.
model = model.to(device)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.00001)
best_acc = 0
epochs = 10


def train(epoch):
    # nn.Module에서는 Train와 Evaluate에서 수행하는 작업을 Switching 해주는 함수 제공
    # Dropout Layer, BatchNorm Layer같은건 Evaluation에서는 필요없음
    model.train()

    train_loss = 0
    correct = 0
    total = 0

    for index, (inputs, targets) in enumerate(train_loader):    # index: 배치번호 / input, targets들은 한 배치안에 있는 모든 input과 target쌍
        # data를 원하는 Device로 배치시키기 (CPU, GPU)
        inputs, targets = inputs.to(device), targets.to(device)

        optimizer.zero_grad()   # for문이 돌때마다 기존에 저장되어있던 gradient 정보값들 0으로 초기화
        
        outputs = model(inputs)
        loss = criterion(outputs, targets)
        loss.backward() # loss 함수를 미분하여 Gradient 계산
        optimizer.step()    # 계산된 w, b값으로 초기화

        train_loss += loss.item()   # loss.item(): 현재 loss값 출력
        _, predicted = outputs.max(1)   # axis:1 방향으로 max값 출력
        # output으로 각 이미지별로 6개항목에 대한 Softmax값이 나오는데
        # _은 가장 큰 Softmax 값을
        # predicted는 가장 큰 Softmax 값의 index를 반환한다

        total += targets.size(0)    # 64 (==batch_size)
        correct += (predicted == targets).sum().item()
        if (index+1) % 20 == 0:
            print(f'[Train] | epoch: {epoch+1}/{epochs} | batch: {index+1}/{len(train_loader)}| loss: {loss.item():.4f} | Acc: {correct / total * 100:.4f}')

def test(epoch):
    global best_acc
    model.eval() 
    test_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for index, (inputs, targets) in enumerate(test_loader):

            inputs, targets = inputs.to(device), targets.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, targets)
         
            test_loss += loss.item()    # 손실이 갖고있는 Scala값 가져오기
            _, predicted = outputs.max(1)   # 어느방향으로 max를 찾을지
            total += targets.size(0)    # 전체 이미지 수
            correct += (predicted == targets).sum().item()  # for문안의 sum으로 누적합 구하고, item으로 predict의 item 가져오기
            
        print(f'[Test] epoch: {epoch+1} loss: {test_loss:.4f} | Acc: {correct / total * 100:.4f}')
        


    # 체크포인트 저장
    acc =100.*correct/total
    if acc > best_acc:
        print('Saving...')
        state = {
            'model': model.state_dict(),
            'acc' : acc,
            'epoch' : epoch,
        }
        if not os.path.isdir('checkpoint'):
            os.mkdir('checkpoint')
        torch.save(state, './checkpoint/ckpt.pth')
        best_acc = acc

if __name__ == "__main__":
    for epoch in range(epochs):
        train(epoch)
        test(epoch)
```

#### 2-3. mobilenetv2_inference.py

```python
import matplotlib.pyplot as plt
import numpy as np
import torchvision.transforms as transforms
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

from PIL import Image
import time
import torchvision
import os
import torch
import torch.nn as nn
from mobilenetv2 import mobilenet_v2


class Archive(Dataset):
    def __init__(self, path, transform=None):
        self.img_name = [f for f in os.listdir(path)]
        self.path = path
        self.imgList = [os.path.join(self.path, i) for i in self.img_name]
        self.transform = transform

    def __len__(self):
        return len(self.imgList)

    def __getitem__(self, idx):
        image = Image.open(self.imgList[idx]).convert('RGB')

        if self.transform is not None:
            image = self.transform(image)
        
        return image

def imshow(inp, title=None):
    inp = inp.numpy().transpose((1, 2, 0))  # shape transpose (왼쪽으로 shift)
    mean = np.array([0.5, 0.5, 0.5])
    std = np.array([0.5, 0.5, 0.5])
    inp = std * inp + mean
    inp = np.clip(inp, 0, 1)
    plt.imshow(inp)

    if title is not None:
        plt.title(title)
    plt.pause(0.001)

def visualize_model(model, num_images=12):
    was_training = model.training   # Training 중인지 아닌지 Bool 반환
    model.eval()
    images_so_far = 0
    fig = plt.figure()

    with torch.no_grad():
        for i, inputs in enumerate(pred_loader):
            inputs = inputs.to(device)
            
            outputs = model(inputs)  # 개별 사진마다 6개항목에 대한 softmax값이 나오는데 여기서
            _, preds = torch.max(outputs, 1)
            # _에는 가장 큰 softmax 값이 나오고
            # preds에는 가장 큰 애의 index가 나옴
            
            for j in range(inputs.size()[0]):   # (64, 3, 224, 224) 즉, 배치사이즈만큼 돌아감
                images_so_far += 1
                plt.figure(figsize=(20, 20))
                ax = plt.subplot(num_images//2, 2, images_so_far)
                ax.axis('off')
                ax.set_title('predicted: {}'.format(classes[preds[j]])) # 예측한 애들을 이제 Class 명으로 변환
                imshow(inputs.cpu().data[j])

                if images_so_far == num_images:
                    model.train(mode=was_training)  # model.train(mode=False) == model.eval()
                    return
        model.train(mode=was_training)

if __name__ == "__main__":

    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    train_path = './dataset/seg_train/seg_train'
    pred_path = './dataset/seg_pred/seg_pred'
    classes = sorted(os.listdir(train_path))

    pred_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize((.5, .5, .5), (.5, .5, .5))
    ])

    pred_dataset = Archive(pred_path, transform=pred_transform) # Model에 Data 넣기 전에, model에 맞게 데이터 Transform
    pred_loader = DataLoader(pred_dataset, batch_size=64, shuffle=True, num_workers=6, pin_memory=True) # 데이터 묶기

    model = mobilenet_v2(False)
    model.classifier = nn.Linear(model.classifier.in_features, len(classes)).to(device) # 1000 -> 6

    checkpoint = torch.load('checkpoint/ckpt.pth')
    model.load_state_dict(checkpoint['model'])  # weight와 모델 가져오기

    model = model.to(device)
    visualize_model(model)
```

---

### 3. 결과

> 빠르게 학습되어, 10 epoch만에 93%의 Accuracy를 달성

- epoch: 1

![image-20220725104204282](https://raw.githubusercontent.com/ech97/save-image-repo/master/image/image-20220725104204282.png)

- epoch: 10

![image-20220725104154785](https://raw.githubusercontent.com/ech97/save-image-repo/master/image/image-20220725104154785.png)
