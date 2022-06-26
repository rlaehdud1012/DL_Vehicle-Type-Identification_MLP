# DL_Vehicle-Type-Identification_MLP

### 사용한 라이브러리(전처리)

1. import pandas as pd
2. import numpy as np
3. import tensorflow as tf
4. import matplotlib.pyplot as plt
5. import os
6. import cv2
7. import glob
8. from PIL import Image
9. from sklearn.model_selection import train_test_split
10. from tensorflow import keras
11. from tensorflow.keras.utils import to_categorical, plot_model
12. import pickle

### 사용한 라이브러리(MLP)
1. from tensorflow.keras.models import Sequential
2. from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten, LeakyReLU

### 목차
- 서론
  분석 목적
  데이터 소개
- 본론
  분석 프로세스
  데이터 전처리
  데이터 분석
- 결론
  분석 결과
  
### 분석 목적
주유소, 주차장, 도로 등의 CCTV에 촬영된 이미지로부터 차종 식별

### 데이터 소개
- AI HUB 사이트에 있는 자동차 차종/연식/번호판 인식용 영상
- 부천시 내 CCTV 및 별도로 설치한 카메라로부터 약 2,189시간의 영상 수집
- 영상 데이터를 이미지로 추출하여, 가공작업을 진행한 데이터

### 분석 프로세스
1. 데이터 수집
2. 딥러닝 네트워크 선정
3. 모델 구축
4. 모델 적합

### 데이터 전처리
1. 이미지 데이터셋 만드는 과정
- 데이터 수집(전체 데이터 453,276장)
- 이미지 불러오기(Pilow 라이브러리 사용)
- 이미지 사이즈 조정(128x128 픽셀; 데이터의 크기가 매우 커 후반부에 갈수록 'MemoryError' 발생하여 픽셀을 낮추는 작업을 진행)
- 데이터셋 분할(train_test_split 함수 사용; test_size 0.2 / stratify=y 로 설정)
- 데이터셋 저장(pickle 모듈 안에 dump 함수를 통해서 데이터셋을 저장/불러오기 진행)

2. 정규화 및 One-hot 벡터 전환
- 0~1 사이의 실수값으로 변환
X_train = X_train.reshape(-1,128,128,1).astype('float32') / 255
- 11개의 범주형 자료를 one-hot 벡터로 변환
y_train = to_categorical(y_train)

### MLP(Multilayer Perceptron)
1. MLP 모형 코드

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Activation, Dropout, Flatten

model = Sequential()
model.add(Dense(128, input_dim = num_features, activation='relu', kernel_initializer='he_normal')) # 1 layer / 모수 : 128 x( (128x128) +1 ) = 2097280
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal')) # 2 layer / 모수 : 128 x(128+1) = 16512
model.add(Dropout(0.5))
model.add(Dense(128, activation='relu', kernel_initializer='he_normal')) # 3 layer / 모수 : 128 x(128+1) = 16512
model.add(Dropout(0.5))
model.add(Dense(11, activation='softmax'))
model.summary()

model.compile(loss='catgorical_crossentropy', optimizer='adam', metrics=['accuracy', tf.keras.metrics.Recall()]), history=model.fit(X_train, y_train, batch_size=64, epochs=100, validation_split=0.1)

2. 성능 측정
* Layers : 128-128-128
* Activation : Relu / LeakyRelu / SELU
* Optimizer : Adam / RMSprop
* Regularizer : Dropout 0.5
- Train Loss / Accuracy / Recall 값이 가장 큰 모수를 최적화된 모형으로 선정하였습니다.

### 아쉬운 점
- 데이터 전처리과정에서 시간소요가 커 다양한 하이퍼파라미터를 적용하지 못해 아쉬웠습니다.
