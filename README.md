## LICA-project

### 설명
2~4개의 성냥으로 12개의 모양을 만듬. (정답 성냥)  
실험자는 정답 성냥을 본 뒤 모양과 방향을 재현하고, 그 이미지를 모델에 넣어 정답과 비교 후 모양만 맞을 경우 0.5, 방향만 맞을 경우 0.5, 둘다 맞을 경우 1점을 부여하도록 함.

***

### 정답 성냥 예시
![image](https://user-images.githubusercontent.com/43367868/112931107-77ea7080-9156-11eb-91b5-2f49faa3c421.png)
*** 

### 모델 구성도
![image](https://user-images.githubusercontent.com/43367868/113096926-13501400-9231-11eb-94bc-478e6147d991.png)  
***

### 모델 설명
- 데이터 전처리  
원본이미지를 좌우로 20도씩 회전, 256x256 크기로 crop 함. 이때 성냥의 빨간 머리가 잘리지 않도록 하였으며 잘린 이미지는 사용하지 않음. --> 모양 이미지  
성냥의 방향을 판단하기 위해 전처리한 이미지에서 빨간색만 추출한 이미지도 함께 학습시킴. --> 방향 이미지

- 모델
CNN 모델 사용.   
모양과 방향 둘 다 따로 판단해야하기 떄문에 모양이미지 layer (data_net), 방향이미지 layer(red_net) 를 따로 만든 뒤  
각 layer를 통과한 이미지를 다시 2층짜리 convolution layer에 통과시켜 합침(combine_net).  

- 학습
총 12000개의 data set을 학습용 : 11000개, 테스트용 : 1000개로 나누어 학습시킴.  
정답 여부는 BCELoss 를 사용하여 이진 모델로 판단



