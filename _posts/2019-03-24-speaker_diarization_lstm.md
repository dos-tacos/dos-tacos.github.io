---
layout: single
title:  "About Speaker Diarization with LSTM"
header:
  teaser: "images/unsplash-gallery-image-2-th.jpg"
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: "images/unsplash-gallery-image-2-th.jpg"
categories: 
  - paper review
tags:
  - speaker diarization
  - SAD
  - LSTM
  - d-vector
toc: true
toc_label: "목차"
toc_icon: "cog"

author: daydrill
---



# INTRO
* Speaker diarization는 입력 된 오디오 스트림을 화자에 따라 균질 세그먼트로 분할하는 프로세스입니다. 이는 다화자 환경에서 “who spoke when"에 대한 질문에 답을 찾는 것을 의미합니다. 특히, 이를 통해 얻을 수 있는 화자 경계(speaker boundary)는 음성인식(ASR) 정확도의 상당하게 향상시킬 수 있는 가능성을 가지고 있습니다.
* 최근의 통상적인 SD 시스템은 보통 다음 4가지로 구성됩니다.
    * (1) Speech segmentation : 오디오 스트림을 비음성 구간을 필터링하고, 단일 화자로 구성된 작은 세그멘트로 잘게 나눔.
    * (2) Audio embedding extraction : MFCCs, speaker factors, i-vectors와 같은 특정 특징이 분류 된 섹션에서 추출되는 오디오 임베딩 추출.
    *  (3) Clustering : 화자의 수가 결정되고, 추출 된 오디오 임베딩을 이용해 각 화자에 해당하는 군집을 생성.
    * (4) Resegmentation : 클러스터링 결과가 최종적으로 결과를 더 잘 나타 내기 위해 다시 한번 정제.
* 최근에는 신경망 기반 오디오 임베딩 (d-vector)이 화자 검증(Speaker verification)에 널리 활용되고 있으며, 이전의 i-vector 기반의 state-of-the-art(SOTA)를 능가하게 되었습니다. 그러나 대부분의 응용은 텍스트에 의존적인(text-dependent) 화자 검증(Speaker verification)에 속하며, 여기서 화자 임베딩은 특정하게 탐지 된 키워드에서 추출됩니다. 반대로, SD는 임의의 음성에 작용하는 텍스트에 독립적인(text-independent) 임베딩 방법을 필요로 합니다.
* 이 논문에서는 우리는 SD를 위한 텍스트에 독립적인 d-vector 기반 접근법에 대해서 다룹니다. 우리는 'Generalized end-to-end loss for speaker verification (2017)'의 작업을 활용하여 LSTM 기반의 텍스트에 독립적인 화자 검증 모델을 학습한 다음 이 모델을 비모수적(non parametric) spectral clustering 알고리즘과 결합하여 SOTA SD 시스템을 제안합니다. 기존의 FFNN기반의 연구는 몇 있었지만, 이 연구가 LSTM 기반의 d-vector 임베딩과 스펙트럼 클러스터링을 결합한 최초의 연구입니다. 뿐만아니라 스펙트럼 클러스터링 알고리즘의 일부로 affinity matrix 노이즈 감소에 효과적으로 작용하여 SD 시스템에 기여하는 affinity matrix refinement 단계를 제안합니다.


# Diarization with d-vectors
* 'Generalized end-to-end loss for speaker verification (2017)’에서는 화자 검증을 위해 LSTM 기반의 화자 임베딩을 제안했습니다. 여기서 모델은 대용량 자유발화 코퍼스에서 고정 길이의 세그먼트를 통해 학습이 됩니다. 저자는 이러한 네트워크에 의해 생성 된 d-vector 임베딩이 일반적으로 등록-검증(enrollment-verification) 2 단계 어플리케이션에서 i-vector를 훨씬 능가한다는 것을 보여주었습니다. 이제 이 모델이 우리의 목적인 SD를 위해 수정 될 수있는 방법에 대해서 살펴보도록 하겠습니다.

<center><img src="/images/daydrill/2019-03-24/1.png" height="400"></center>

* SD시스템 플로우차트는 위의 그림과 같습니다. 이 시스템에서 오디오 신호는 먼저 폭 25ms와 단계 10ms의 프레임으로 변환되고, 네트워크의 입력을 위해 40차 log-mel-filterbank로 피쳐를 뽑게 됩니다. 이 프레임에 고정 길이의 슬라이딩 윈도우를 만들고 각 윈도우에서 LSTM 네트워크를 실행합니다. LSTM의 마지막 프레임 출력은이 슬라이딩 윈도우의 d-vector representation으로 사용됩니다.
* VAD (Voice Activity Detention)를 사용하여 오디오에서 음성 세그먼트를 결정합니다. 음성 세그먼트는 최대 세그먼트 길이 제한 (예 : 실험에서 400ms)을 사용하여 더 작은 겹치지 않는 세그먼트로 나누어집니다. 여기서 이 세그먼트는 SD의 시간 해상도를 결정합니다. 그리고 각각의 세그먼트에 대해, 대응하는 d-vector가 먼저 L2 정규화되고,이어서 세그먼트의 임베딩을 형성하기 위해 평균화됩니다. 위의 과정은 임의 길이의 오디오 입력을 고정 길이 임베딩 시퀀스로 줄이는 역할을 합니다. 이제는 고유 한 스피커의 수를 확인하고 오디오의 각 부분을 특정 스피커에 할당하기 위해 이러한 포함에 클러스터링 알고리즘을 적용 할 수 있습니다.


# CLUSTERING
* 이 섹션에서는 우리의 분산 시스템에 통합 된 네 가지 클러스터링 알고리즘을 소개합니다. 우리는 스펙트럼 오프라인 클러스터링 알고리즘에 특별한 초점을 두었습니다.이 알고리즘은 실험을 통해 대체 접근법을 크게 뛰어 넘었습니다.
* 클러스터링 알고리즘은 런타임 대기 시간에 따라 두 가지 범주로 구분할 수 있습니다
* 온라인 클러스터링 : 향후 세그먼트를 보지 않고도 세그먼트를 사용할 수있는 경우 스피커 레이블이 즉시 생성
* 오프라인 클러스터링 : 스피커 레이블이 모든 세그먼트의 임베딩이 사용가능할 때 생성
* 오프라인 클러스터링 알고리즘은 일반적으로 오프라인 설정에서 사용할 수있는 추가 컨텍스트 정보로 인해 온라인 클러스터링 알고리즘보다 성능이 우수합니다. 또한 최종 세분화 단계는 오프라인 설정에서만 적용 할 수 있습니다. 그럼에도 불구하고 온라인과 오프라인 사이의 선택은 주로 시스템의 배포를 목적으로하는 응용 프로그램의 성격에 달려 있습니다. 예를 들어 라이브 비디오 분석과 같은 대기 시간에 민감한 응용 프로그램은 일반적으로 시스템을 온라인 클러스터링 알고리즘으로 제한합니다.


# DISCUSSION
* 음성 데이터 분석은 매우 어려운 문제 영역이며 K-Means와 같은 기존의 클러스터링 알고리즘은 종종 제대로 수행되지 않습니다. 이는 다음을 포함하는 음성 데이터에 내재 된 많은 불행한 속성 때문입니다.
* Non-Gaussian Distributions : 음성 데이터는 종종 Non-Gaussian입니다. 이러한 환경에서 각 클러스터의 센트로이드(K-Means 클러스터링의 중심)은 충분한 표현방법이 아닙니다.
* Cluster Imbalance : 음성 데이터에서 어떤 사람은 말을 많이 하지만, 어떤 사람은 말을 적게 하는 경우가 있습니다. 이 때, K-Means는 대용량 클러스터를 여러 개의 작은 클러스터로 잘못 분류 할 수 있습니다.
* Hierarchical Structure : 성별, 연령, 악센트 등에 따라 여러 그룹으로 구분됩니다. 이 구조는 남성과 여성 스피커의 차이가 두 명의 남성 스피커의 차이보다 훨씬 크기 때문에 문제가 될 수 있습니다. 이는 K-Means가 그룹에 해당하는 클러스터와 별개의 스피커에 해당하는 집단을 구분하는 것을 어렵게 만듭니다. 실제로 이것은 K-Means가 남성 스피커에 해당하는 모든 내장을 하나의 클러스터로 잘못 클러스터링하고 여성 스피커에 해당하는 모든 임베딩을 다른 클러스터로 잘못 클러스터링합니다.
* 이러한 속성으로 인해 발생하는 문제는 K-Means 클러스터링에 국한되지 않고 대부분의 파라 메트릭 클러스터링 알고리즘에 특유합니다. 다행스럽게도 이러한 문제는 asspectral 클러스터링과 같은 비모수 적 연결 기반 클러스터링 알고리즘을 사용하여 완화 할 수 있습니다.

