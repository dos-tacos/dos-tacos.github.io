---
layout: single
header:
  teaser: images/lynn/190519/header.png
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image:  images/lynn/190519/header.png
excerpt: "\"Attention is all you need\" 페이퍼 리뷰: sequence transduction 모델의 강자로 떠오르고 있는 트랜스포머 모델에 대해 알아봅니다."
title:  "Sequence prediction(6): Transformer network"
categories: 
  - paper review
tags:
  - google
  - transformer
  - sequence transduction
author: Lynn Hong
toc: true
toc_label: "목차"
toc_icon: "cog"
---

# 들어가며

- 참고 자료: [The Transformer – Attention is all you need.](https://mchromiak.github.io/articles/2017/Sep/12/Transformer-Attention-is-all-you-need/#.XGvUqOgza7e)

# Repositories
- [Official Transformer translation model](https://github.com/tensorflow/models/tree/master/official/transformer)
- [Attention is all you need: A Pytorch Implementation](https://github.com/jadore801120/attention-is-all-you-need-pytorch)
- [PyTorch implementation of OpenAI's Finetuned Transformer Language Model](https://github.com/huggingface/pytorch-openai-transformer-lm)

# Abstract

- 현재의 지배적인 시퀀스 transduction 모델들은 인코더와 디코더를 포함하는 복잡한 recurrent 또는 convolutional 신경망을(RNN, CNN) 기반으로 함
- 또한 최고 성능의 모델들은 attention 매커니즘을 통해 인코더와 디코더를 연결
- **저자들은 '트랜스포머(Transformer)'라는 새롭고 간단한 네트워크 구조를 제안. 이는 오로지 attention 매커니즘에 기반하고 있으며 recurrence와 convolution 자체를 완전히 제거해 버린 형태**
- 두 가지 기계 번역 task에 대한 실험은 이러한 모델들이 품질 측면에서 우수하고 병렬 처리가 가능하며 훈련 시간이 훨씬 적음을 보여줌
  - WMT 2014 영어-독일어 번역 task에 대해 28.4 BLEU를 달성하여 앙상블을 포함한 기존 최고의 결과를 2 BLEU 이상으로 향상
  - WMT 2014 영어-프랑스어 번역 task에서 8개의 GPU를 이용해 3.5일 동안 훈련한 후, 단독 모델로 최고 점수인 41.8 BLEU를 기록
- 대량의 제한적인 학습 데이터에서의 영어 문장 구성성분 분석(constituency parsing)에 이 모델을 성공적으로 적용함으로써 트랜스포머가 다른 task들에도 일반화하기 용이하다는 것을 증명

# 1. Introduction

- 특히 언어 모델링 및 기계 번역과 같은 시퀀스 모델링 및 변환 문제에서 RNN(Recurrent Neural Network), LSTM(Long short-term memory) 그리고 Gated recurrent 신경망 등은 최고 수준의 접근 방식으로 확고하게 자리잡음
  - 이후 recurrent 언어 모델과 인코더-디코더 구조의 지평을 넓혀 나가려는 수많은 노력이 계속되어 옴
- RNN은 일반적으로 input과 output 시퀀스의 각 기호 위치를 따라 팩터(factor) 계산이 이루어짐
- 위치들을 계산 시간의 각 단계에 align(정렬)하면, 직전 hidden state인 $$h_t-1$$과 위치 $$t$$를 input으로 받아 hidden states의 시퀀스인 $$h_t$$를 생성
- 이러한 본질적으로 연속적인 특성은 학습 샘플들 내에서 병렬처리를 불가능하게 함
  - 메모리 제약 조건들이 샘플 간의 배치(batch)를 제한하기 때문에 이는 시퀀스의 길이가 길어질 때 매우 중요한 문제
- 최근의 연구에서는 factorization 트릭과 conditional computation을 통해 계산 효율화 측면에서의 의미있는 성과를 거두었음
  - 후자의 경우 모델 성능 또한 향상시킨 바 있음
  - 그러나 순차적(sequential) 계산 방식의 근본적인 제약은 여전히 남아있는 상태
- attention 매커니즘은 다양한 task에서 강력한 시퀀스 모델링 및 변환 모델의 필수적인 부분이 되었음
  - 입력 또는 출력 시퀀스에서의 **거리에 관계없이 종속성을 모델링**할 수 있음
  - 그러나 이러한 attention 매커니즘은 몇 가지 경우를 제외하고는 모두 recurrent 네트워크와의 결합으로 함께 사용
- 이 연구에서는 **반복(recurrence)을 삼가고, 대신 입출력 간에 전역적인 종속성을 끌어내기 위해 attention 메커니즘에 전적으로 의존하는 '트랜스포머'라는 모델 구조를 제안**
  - '트랜스포머'는 훨씬 더 많은 병렬화를 허용
  - 8개의 P100 GPU에서 12시간이라는 짧은 시간 동안 학습된 후 새로운 최고 수준의 번역 성능에 도달

# 2. Background

- 순차적인(sequential) 계산을 감소시킨다는 목표는 또한 다른 많은 모델들을 낳은 토대가 되었음
  - ex. 기본 빌딩 블록으로 CNN을 사용하는 Extended Neural GPU, ByteNet, ConvS2S 등
  - 모든 입력 및 출력 위치에 대해 hidden representation을 병렬로 계산
- 이러한 모델들에서 두 개의 임의의 입력 또는 출력 위치 신호를 연관짓는 데 필요한 연산 수는 위치들 간의 거리에 따라 증가
  - ConvS2S의 경우 선형적으로, ByteNet의 경우 대수적으로 증가
  - 이로 인해 먼 위치 사이의 의존성을 배우기가 더 어려워짐
- '트랜스포머'에서는 이 연산량이 일정한 횟수로 줄어듬
  - 3.2 절에서 설명될 Multi-Head Attention로 대응한 효과
  - attention 가중치 위치를 평균냄으로써 효과적인 해결책 감소를 비용으로 지불
- 'self-attention'('intra-attention')은 시퀀스의 representation을 계산하기 위해 단일 시퀀스 내의 각기 다른 위치들의 연관성을 파악하는 attention 메커니즘의 일종
  - self-attention는 기계 독해(reading comprehension), 추상적 요약(abstract summarization), 텍스트 함의 인식(textual entailment) 및 task 독립적인 문장 representation 학습 등 다양한 task에서 성공적으로 사용됨
- end-to-end 메모리 네트워크는 순서 조정 반복(sequence aligned recurrence) 대신 반복적인 attention 메커니즘을 기반으로 함
  - 간단한 언어 질문 응답 및 언어 모델링 task에서 좋은 결과를 보이는 것으로 나타남
- 그러나 **'트랜스포머'는 순서 조정 RNN이나 컨볼루션을 사용하지 않고 입력 및 출력의 표현을 계산하기 위해 전적으로 self-attention에 의존하는 최초의 transduction 모델** 
  - 다음 섹션에서 우리는 '트랜스포머'를 설명하고 self-attention을 사용한 이유를 설명
  - 또한 다른 모델들에 비교하여 장점을 논의

# 3. Model Architecture

- 대부분의 경쟁적인 뉴럴 시퀀스 transduction 모델은 인코더-디코더 구조 [5, 2, 29]를 가지고 있음
- 여기서 인코더는 기호 표현 $$(x_1, ..., x_n)$$의 입력 시퀀스를 연속 표현 $$z = (z_1, ..., z_n)$$의 시퀀스에 매핑
- $$z$$가 주어지면, 디코더는 한 번에 하나의 요소의 기호의 출력 시퀀스 $$(y_1, ..., y_m)$$를 생성
  > 역주: x -> 인코더 -> z -> 디코더 -> y
- 각 단계에서 모델은 auto-regressive [^10]하며, 다음 생성시 이전에 생성된 기호를 추가 입력으로 받아들임
  > 역주: auto-regressive(AR)은 자기 자신을 입력으로 하여 자신을 예측하는 모델



- 트랜스포머 모델은 Figure 1의 왼쪽과 오른쪽 절반에 각각 표시된 인코더와 디코더 모두에 대해 겹쳐 쌓인 self-attention 및 point-wise fully-connected 레이어를 사용하여 이 전체 아키텍처를 따름

<img src="/images/lynn/190519/1.PNG" width="500px;" style="text-align: center;"/>

<em>Figure 1. The Transformer - 모델 구조</em>


## 3.1 Encoder and Decoder Stacks

### Encoder
- 인코더는 $$N=6$$의 동일한 레이어를 쌓아 구성
- 각 레이어에는 두 개의 하위 레이어(sub-layer)가 있음
  - 1) multi-head self-attention 메커니즘
  - 2) 단순한 형태의 fully-connected FNN
- 두 하위 레이어에 각각 residual connection[^11]을 사용하고 그 후에 레이어 정규화(layer normalization) [^1]를 사용
- 즉, 각 하위 레이어의 출력은 $$LayerNorm(x + Sublayer(x))$$, 여기서 $$Sublayer(x)$$는 하위 레이어 자체에 의해 구현되는 함수
- 이 residual connection을 용이하게 하기 위해 모델의 모든 하위 레이어과 임베딩 레이어는 $$d_{model} = 512$$ 차원의 출력을 생성
> 역주: residual connetion(skip-connection)은 input을 그대로 output으로 전달하는 것으로 연결 자체를 skip한다는 개념. Resnet에서 사용되었고, 위의 residual connection의 참고문헌도 Resnet 논문임

> 역주: layer normalization은 아래와 같은 공변량 변화를 극복하기 위한 것
>
> 공변량 변화(covariate shift): 주어진 입력에 대한 출력 생성 규칙 자체는 학습 시와 test 시에 달라지지 않는데, 미묘하게 입력 데이터(공변량)의 분포가 달라지는 상황. 레이어를 쌓는 DNN의 경우 바로 직전 레이어의 분포를 참고하기 때문에 어느 순간 파라미터가 엉뚱하게 업데이트 될 수 있음. 이는 층이 깊어질수록 더 심해지는 현상
>
> <img src="/images/lynn/190519/4.PNG" width="100%;" style="text-align: center;"/>
>
> <img src="/images/lynn/190519/3.PNG" width="100%;" style="text-align: center;"/> 
>
> <img src="/images/lynn/190519/5.PNG" width="100%;" style="text-align: center;"/>
>
> LN은 동일 층의 뉴런 간 정규화. mini-batch 샘플 간 의존 관계가 없음. 온라인 및 RNN으로 확장이 가능
>
> <img src="/images/lynn/190519/6.PNG" width="100%;" style="text-align: center;"/> [^20]

[^20]: https://www.slideshare.net/ssuser06e0c5/normalization-72539464

### Decoder
- 디코더 또한 $$N = 6$$의 동일한 레이어를 쌓아 구성
- 디코더는 각 인코더 층의 두 개의 하위 레이어(sub-layer) 외에도 세번째의 하위 레이어를 삽입. 이는 인코더 스택의 출력에 대해 multi-head self-attention를 수행
- 인코더와 마찬가지로 각 하위 레이어 주위에 residual connection을 사용하고 그 다음에 레이어 정규화 사용
- 또한 디코더 스택에서의 self-attention 하위 레이어을 수정하여 다음 위치로의 간섭을 방지
  - 이러한 마스킹(masking)은 출력 임베딩이 하나의 위치로 상쇄된다는 사실과 결합. 위치 $$i$$에 대한 예측이 $$i$$보다 작은 위치에서 알려진 출력에만 의존할 수 있도록 보장
  > 역주: 마스킹(masking)은 위치 $$i$$ 이후의 것에 미리 attention을 주지 못하게 하는 것. 이전 값들만 이용해 예측

## 3.2 Attention
- attention 함수는 query, key-value 쌍을 출력에 매핑하는 것
  - 여기서 query, key, value 및 출력은 모두 벡터
- 출력(output)은 value의 가중 합으로 계산. 각 값에 할당된 가중치는 해당 키와의 질의의 호환성(compatibility) 함수에 의해 계산

### 3.2.1 Scaled Dot-Product Attention

<img src="/images/lynn/190519/2.PNG" width="600px;" style="text-align: center;"/>

<em>(왼쪽) Scaled Dot-Product Attention. (오른쪽) 병렬적으로 돌아가는 여러 attention 레이어로 구성된 Multi-Head Attention</em>

- 저자들은 그들의 특별한 attention을 "Scaled Dot-Product Attention"(Figure 2)이라고 지칭
  - 입력(input)은 차원 $$d_k$$의 query, key와 차원 $$d_v$$의 value로 구성
- 모든 키에 대해 쿼리의 내적(dot product)을 계산하고 각 값을 $$\sqrt{d_k}$$로 나눔. 그리고 value에 대한 가중치를 얻기 위해 소프트맥스 함수를 적용

- 실제로 행렬 $$Q$$로 함께 묶인(packed together) 일련의 질의들에 대한 attention 함수를 동시에 계산
- key와 value는 또한 행렬 $$K$$와 $$V$$로 함께 묶여 있음
- 출력(output)의 행렬은 다음과 같이 계산 : 

$$
Attention(Q, K, V) = softmax(\frac{QK^T}{\sqrt{d_k}})V \tag{1}
$$

- 가장 일반적으로 사용되는 두 가지 attention 함수는 addictive attention[^2], dot-product(multi-plicative) attention
  - dot-product attention은 $$\frac{1}{\sqrt{d_k}}$$, 즉 스케일링 팩터를 제외하고는 본 연구의 알고리즘과 동일
  - addictive attention는 단일 hidden 레이어를 갖는 FNN를 사용하여 호환성 함수를 계산
- 두 가지가 이론적 복잡성이 비슷하지만, dot-product attention가 실제로 훨씬 빠르고 공간 효율적
  - 고도로 최적화된 행렬 곱셈 코드를 사용하여 구현 될 수 있기 때문
- $$d_k$$의 작은 값에 대해 두 메커니즘은 유사하게 수행되지만, addictive attention은 $$d_k$$의 더 큰 값에 대해 스케일링하지 않고 dot-product attention을 능가[^3]
- 저자들은 $$d_k$$의 큰 값에 대해 내적 값들이 크기가 커져 소프트맥스 함수를 매우 작은 기울기 4를 갖는 영역으로 밀어 넣는다고 의심





[^1]:  Jimmy Lei Ba, Jamie Ryan Kiros, and Geoffrey E Hinton. Layer normalization. arXiv preprint arXiv:1607.06450, 2016.
[^2]: Dzmitry Bahdanau, Kyunghyun Cho, and Yoshua Bengio. Neural machine translation by Jjointly learning to align and translate. CoRR, abs/1409.0473, 2014.
[^3]: Denny Britz, Anna Goldie, Minh-Thang Luong, and Quoc V. Le. Massive exploration of neural machine translation architectures. CoRR, abs/1703.03906, 2017.
[^9]: Alex Graves. Generating sequences with recurrent neural networks. arXiv preprint arXiv:1308.0850, 2013.
[^11]: Sepp Hochreiter, Yoshua Bengio, Paolo Frasconi, and Jürgen Schmidhuber. Gradient flow in recurrent nets: the difficulty of learning long-term dependencies, 2001.
[^17]:  Diederik Kingma and Jimmy Ba. Adam: A method for stochastic optimization. In ICLR, 2015.
[^18]: Oleksii Kuchaiev and Boris Ginsburg. Factorization tricks for LSTM networks. arXiv preprint arXiv:1703.10722, 2017.



### 3.2.2 Multi-Head Attention

- $$d_{model}$$-dimensional keys, values, queries를 이용하여 단일 attention 함수를 수행하는 대신, 우리는 학습된 선형 프로젝션(projection)을 $$d_k$$, $$d_k$$, $$d_v$$ 차원으로 각각 선형적으로 투영하는 것이 유리하다는 것을 발견
- 이러한 투영된 버전의 queries, keys 및 values에서 병렬로 attention 함수를 수행하여 $$d_v$$ 차원 출력 값을 산출
- 이것들은 연결되고 다시 투영되어 Figure 2에 묘사된 바와 같이 최종 값이 됨

- multi-head self-attention은 모델이 서로 다른 위치의 다른 표현 하위 공간의 정보에 결합적으로 주의를 기울일 수 있게 함
  - 단일 attention 헤드에서는, 평균화(averaging)가 이것을 억제

$$
head_i = Attention(QW_i^Q, KW_i^K, VW_i^V) 
$$

일 때,

$$
MultiHead(Q, K, V) = Concat(head_1, ..., head_h)W^O
$$

### 3.2.3 Applications of Attention in our Model

- 트랜스포머는 세 가지 다른 방법으로 multi-head self-attention을 사용

- "인코더 디코더 attention" 계층에서 쿼리는 이전 디코더 계층에서 나오고 메모리 key와 value은 인코더의 출력에서 나옴. 이를 통해 디코더의 모든 위치가 입력 시퀀스의 모든 위치에 주의를 기울일 수 있음
  - 이것은 [38, 2, 9]와 같은 seq2seq 모델에서 일반적인 인코더-디코더 attention 메커니즘을 모방

- 인코더는 self-attention 레이어를 포함. self-attention 레이어에서 모든 키, 값 및 쿼리는 동일한 장소에서 나옴
  - 이 경우 인코더의 이전 레이어 출력
  - 인코더의 각 위치는 인코더의 이전 층의 모든 위치에 주의를 기울일 수 있음

- 유사하게, 디코더 내의 self-attention 레이어는 디코더 내의 각 위치가 디코더의 모든 위치에 그 위치까지 그리고 그 위치를 포함하도록 허용
- auto-regressive 특성을 유지하기 위해서는 디코더에서 왼쪽 정보 흐름을 방지해야 함
- 우리는 잘못된 연결에 해당하는 소프트맥스의 입력에서 모든 값을 마스킹(-로 설정)함으로써 스케일링된 dot-product attention의 내부를 구현
  - Figure 2를 참조


## 3.3 Position-wise Feed-Forward Network

- attention 하위 레이어 외에도, 인코더 및 디코더의 각 레이어는 완전히 연결된 FNN을 포함하며, 이는 각 위치에 개별적으로 동일하게 적용
  - 이것은 그 사이에 RELU 활성화를 갖는 두 개의 선형 변환으로 구성

$$
FNN(x) = max(0, xW_1 + b_1)W_2 + b_2 \tag{2}
$$

- 선형 변환은 서로 다른 위치에 걸쳐 동일하지만 레이어에서 레이어까지 다른 매개 변수를 사용
  - 이것을 설명하는 또 다른 방법은 커널 크기 1을 가진 두 개의 컨볼루션
  - 입력과 출력의 차원은 $$d_{model} = 512$$
  - 내부층은 차원 $$d_{ff} = 2048$$

## 3.4 Embeddings and Softmax

- 다른 시퀀스 transduction 모델과 마찬가지로, 우리는 입력 토큰과 출력 토큰을 차원 $$d_{model}$$의 벡터로 변환하기 위해 학습된 임베딩을 사용
- 또한, 우리는 일반적인 학습된 선형 변환과 소프트맥스 함수를 사용하여 디코더 출력을 예측된 다음 토큰 확률로 변환
- 본 모델에서 두 개의 임베딩 레이어와 [30]과 유사한 프리 소프트맥스 선형 변환 사이의 동일한 가중치 행렬을 공유. 이러한 가중치를 $$d_{model}$$로 곱함

## 3.5 Positional Encodin

- 모델에 recurrence와 convolution이 없기 때문에 모델이 시퀀스의 순서를 사용하기 위해서는 시퀀스에서 토큰의 상대적 또는 절대적 위치에 대한 정보를 주입해야 함
- 이를 위해 인코더 및 디코더 스택의 하단의 입력 임베딩에 "위치 인코딩"을 추가
  - 위치 인코딩은 임베딩과 동일한 치수 $$d_{model}$$을 가지므로 두 가지를 합산할 수 있음
  - 위치 부호화에는 학습되고 고정된 여러 가지 선택이 있다[9].

- 본 연구에서는 서로 다른 frequency의 sine, cosine 함수를 사용:
  > 역주: frequency는 한 구간 내에서 완료되는 사이클 개수

수식

- 여기서 pos는 위치이고 나는 치수
  - 즉, 위치 부호화의 각 차원은 정현파에 해당
- 파장은 2에서 10000 · 2까지의 기하학적 진행을 형성
- 저자들은 이 함수가 고정된 offset k에 대해 PEpos + k가 PEpos의 선형 함수로 표현 될 수 있기 때문에 모델이 상대적인 위치에 쉽게 참석할 수 있을 것이라고 가정했기 때문에 선택했다고 함

- 또한 학습된 위치 삽입물 [9]을 대신 사용하여 실험한 결과, 두 버전이 거의 동일한 결과를 산출한다는 것을 발견(표 3 행 (E) 참조)
- 모델이 학습 중에 마주친 것보다 더 긴 시퀀스 길이를 추정할 수 있기 때문에 sine 버전을 선택

# 4. Why Self-Attention

- 이 섹션에서는 하나의 가변 길이의 심볼 표현 시퀀스$$(x_1, ..., x_n)$$를 동일한 길이의 다른 시퀀스$$(z_1, ..., z_n)$$에 매핑하기 위해 일반적으로 사용되는 recurrent 및 convolution 레이어에 대해 self-attention 레이어의 다양한 측면을 비교
- self-attention의 사용을 동기부여하여 세 가지의 데시드라타를 고려

1. 레이어 당 계산의 복잡성
2. 필요한 최소 순차 연산 수로 측정한 바와 같이 병렬화할 수 있는 계산량
3. 네트워크에서의 장거리 의존성 사이의 경로 길이
  - 장거리 의존성을 학습하는 것은 많은 시퀀스 변환 작업에서 핵심적인 과제
  - 이러한 종속성을 학습하는 능력에 영향을 미치는 하나의 핵심 요소는 네트워크에서 전진 및 후진 신호를 통과해야하는 경로의 길이
  - 입력 시퀀스와 출력 시퀀스의 모든 위치 조합 사이의 이러한 경로가 짧을수록 장거리 의존성을 배우는 것이 더 쉬움[12]. 따라서 우리는 또한 서로 다른 레이어 유형으로 구성된 네트워크에서 두 입력 위치와 출력 위치 사이의 최대 경로 길이를 비교

- Table 1에서 언급했듯이, self-attention 레이어는 모든 위치를 일정한 수의 순차 실행 동작과 연결하지만, recurrent 레이어는 O(n) 순차 동작을 필요로 함
- 계산 복잡성 측면에서, 시퀀스 길이 n이 표현 차원 d보다 작을 때 자기 주의 층은 반복 레이어보다 빠르며, 이는 워드 피스 [38] 및 바이트 쌍 [31] 표현과 같은 기계 번역에서 최첨단 모델에서 사용되는 문장 표현의 경우 가장 자주 사용됨
- 매우 긴 시퀀스를 포함하는 작업에 대한 계산 성능을 향상시키기 위해 자기 주의는 입력 시퀀스에서 각 출력 위치를 중심으로 R 크기의 이웃만을 고려
  - 이것은 최대 경로 길이를 O(n/r)로 증가시킬 것
  - 본 연구는 향후 연구과제에서 이러한 접근방법을 더욱 연구할 계획

- 커널 폭 k < n을 갖는 단일 컨볼 루션 레이어는 모든 쌍의 입력 및 출력 위치를 연결하지 않음
- 이렇게 하려면 인접한 커널의 경우 O(n/k) 컨볼루션 레이어의 스택이 필요하거나 확장된 컨볼루션의 경우 O(logk(n))가 필요하며 네트워크의 두 위치 사이의 가장 긴 경로의 길이를 증가시킴
- 컨볼루션 층은 일반적으로 k의 요인에 의해 반복 층보다 더 복잡도가 높음
- 분리 가능한 컨볼 루션 [6]은 복잡성을 O (k · n · d + n · d2)로 상당히 감소
- 그러나 k = n을 가지고도 분리 가능한 컨볼루션의 복잡성은 우리가 모델에서 취하는 접근 방식인 자기 주의층과 점 방향 피드 포워드 레이어의 조합과 같음

- 부수적인 이익으로 self-attention은 더 해석 가능한 모델을 산출할 수 있음
  - 우리는 모델에서 주의 분포를 검사하고 부록에 있는 예를 제시하고 토론
  - 개별 attention 헤드는 다른 ta를 수행하는 법을 분명히 배우는 것뿐만 아니라

# 5. Tranining

## 5.1 Training Data and Batching

- 약 450만 개의 문장 쌍으로 구성된 표준 WMT 2014 영어-독일 데이터 세트를 학습
- 문장은 약 37000 토큰의 공유 소스 대상 어휘를 가진 바이트 쌍 인코딩 [3]을 사용하여 인코딩
- 영어-프랑스어의 경우, 36M 문장과 분할 토큰으로 구성된 상당히 큰 WMT 2014 영어-프랑스 데이터셋을 32000 word-piece 어휘[38]로 사용
- 문장 쌍은 대략적인 순서 길이로 함께 배치
  - 각 학습 batch에는 약 25000개의 소스 토큰과 25000개의 타겟 토큰이 포함된 문장 쌍 세트가 포함

## 5.2 Hardware and Schedule

- 8개의 NVIDIA P100 GPU로 한 기계에 우리의 모델을 학습
- 논문 전반에 걸쳐 설명된 하이퍼 파라미터를 사용하는 기본 모델의 경우 각 학습 단계는 약 0.4초가 소요
- 기본 모델을 총 10만 단계 또는 12시간 동안 학습
- 큰 모델의 경우(표 3의 밑줄에 설명) 스텝타임은 1.0초
  - 큰 모델은 30만 단계(3.5일) 동안 학습

## 5.3 Optimizer

- β1 = 0.9, β2 = 0.98 및 = 10-9의 Adam optimizer [20]를 사용
- 저자들은 다음 공식에 따라 훈련 과정에서 학습률을 변화

수식 3

- 이것은 첫 번째 워밍업 _ 스텝 훈련 단계에 대해 학습 속도를 선형으로 증가시키고 그 후 단계 번호의 역 제곱근에 비례하여 감소시키는 것
  - 우리는 워밍업_스텝 = 4000을 사용

## 5.4 Regularization

- 우리는 훈련 중에 세 가지 유형의 정규화를 사용합니다.
- 잔여 드롭아웃: 각 하위 계층의 출력에 드롭아웃[33]을 적용하고, 하위 계층 입력에 추가하여 정규화한다. 또한, 인코더와 디코더 스택 모두에서 임베딩의 합과 위치 부호화에 드롭 아웃을 적용합니다.기본 모델의 경우 Pdrop = 0.1의 비율을 사용한다.
- 라벨 스무딩: 훈련 중에, 우리는 값 = 0.1 [36]의 라벨 스무딩을 사용했다.모델이 더 확실하지 않은 것을 배우지만 정확성과 BLEU 점수를 향상 시키므로 혼란을 낳습니다.

# 6. Result

## 6.1 Machine Translation

<img src="/images/lynn/190519/7.PNG" width="600px;" style="text-align: center;"/>

- WMT 2014 영독 번역 작업에서 빅 트랜스포머 모델(테이블 2의 빅 트랜스포머(빅)은 이전에 보고된 모델(앙상블 포함)을 2.0 BLEU 이상 앞지르며 28.4의 새로운 최첨단 BLEU 점수를 확립
- 이 모델의 구성은 표 3의 하위 라인에 나열. 8개의 P100 GPU에서 3.5일 동안 훈련
  - 심지어 우리의 기본 모델도 경쟁 모델의 훈련 비용의 일부에서 이전에 발표된 모든 모델과 앙상블을 능가

<img src="/images/lynn/190519/8.PNG" width="600px;" style="text-align: center;"/>

- WMT 2014 영어-프랑스어 번역 작업에서, 우리의 큰 모델은 41.0의 BLEU 점수를 달성
  - 이전에 출판된 모든 단일 모델보다, 이전 최첨단 모델의 훈련 비용의 1/4 미만으로 성능이 뛰어남
  - 영어에서 프랑스어로 훈련된 트랜스포머(빅) 모델은 0.3 대신 drop-out rate Pdrop = 0.1을 사용

- 기본 모형은 10분 간격으로 작성된 마지막 5개의 체크포인트를 평균하여 얻은 단일 모형을 사용
- 큰 모형의 경우 마지막 20개의 체크포인트를 평균화
- 본 연구에서는 4개의 빔 크기와 길이 페널티 α = 0.6 [38]의 빔 탐색을 사용
  - 이러한 하이퍼 파라미터는 개발 세트에 대한 실험 후 선택
- 입력 길이 + 50으로 추론하는 동안 최대 출력 길이를 설정하지만 가능한 경우 일찍 종료 [38].

- 표 2는 우리의 결과를 요약하고 우리의 번역 품질과 교육 비용을 문헌의 다른 모델 아키텍처와 비교
- 본 연구에서는 각 GPU 5의 훈련시간, 사용횟수, 유지된 단일 정밀부유점 용량을 추정하여 모델을 훈련하는데 사용되는 부동점 연산 횟수를 추정

## 6.2 Model Variations

- 트랜스포머의 각 구성 요소의 중요성을 평가하기 위해, 우리는 개발 세트인 뉴스테스트2013에서 영어와 독일어 번역의 성능 변화를 측정하면서, 우리의 기본 모델을 다른 방식으로 변화
- 앞 장에서 설명한 바와 같이 빔 탐색을 사용했지만, 검문소 평균은 없었음
  - 이 결과를 표 3에 제시

- 표 3 행 (A)에서, 우리는 3.2.2 절에서 설명한 바와 같이 계산량을 일정하게 유지하면서 주의 헤드의 수와 주의 키 및 값 치수를 변경
- 싱글 헤드 attention은 0.9 BLEU가 가장 좋은 설정보다 좋지 않지만 품질은 너무 많은 헤드로 떨어짐

- 표 3열 (B)에서 attention 키 크기 dk를 줄이면 모델 품질이 저하된다는 것을 관찰
- 이는 호환성을 결정하는 것이 쉽지 않으며 도트 제품보다 보다 정교한 호환성 기능이 유익할 수 있음을 시사
- 우리는 더 많은 모델들이 더 좋고, drop-out이 과적합을 피하는데 매우 도움이 된다는 것을 행(C)과 (D)에서 관찰
- 행 (E)에서 우리는 정현파 위치 부호화를 학습된 위치 삽입물 [9]로 대체하고 기본 모델과 거의 동일한 결과를 관찰

# Reference
- [Auto Regressive Models](https://ratsgo.github.io/generative%20model/2018/01/31/AR/)
- [What are “residual connections” in RNNs?](https://stats.stackexchange.com/questions/321054/what-are-residual-connections-in-rnns)
- [Attention is all you need paper 뽀개기](https://pozalabs.github.io/transformer)
- [Normalization 방법](https://www.slideshare.net/ssuser06e0c5/normalization-72539464)

