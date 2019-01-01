---
layout: single
title:  "Financial news predicts stock market volatility better than close price review (KR)"
header:
  teaser: "images/mason/2018-12-17/img.jpeg"
  overlay_color: "#000"
  overlay_filter: "0.5"
  overlay_image: images/mason/2018-12-17/img2.jpg
  caption: "Photo credit: [**Unsplash**](https://unsplash.com)"
categories: 
  - Paper Review
tags:
  - Stock Market
  - NLP
  - Time Series
author: mason seo

toc: true
toc_label: "목차"
toc_icon: "cog"

---

## Abstract

1. 금융시장으로부터 얻은 시계열 데이터의 변동은 금융시스템에서 얻은 **정량적 정보**와
다양한 형태로 유입되는 기본정보에 대한 **정성적 정보**의 영향을 받습니다.
2. 실험적 결과를 나타내기 위해 입력데이터로 주식 , 주식지수 , 뉴스소스를 입력데이터로 사용하면서 [LDA(Latent Dirichlet Allocation)](https://en.wikipedia.org/wiki/Latent_Dirichlet_allocation)기계학습 모델을 구성하고
[Naïve Bayes](https://en.wikipedia.org/wiki/Naive_Bayes_classifier)을 이용해 이동방향을 예측합니다.
3. 실험해본 결과 위의 입력데이터를 이용한 [Closing price(종가)](https://www.investopedia.com/terms/c/closingprice.asp)예측 정확도는 49%이며 [Volatility(변동성)](https://www.sciencedirect.com/topics/economics-econometrics-and-finance/volatility)의 방향 예측정확도는 56%로 도출되며 해당 논문은 금융 뉴스를 기계 학습 입력으로 사용할 때 변동성 이동이 자산 가격 변동보다 예측 가능하며 변동성을 계량화 이용 될 수 있다고 결론 내립니다.


## 1.Introduction

* 최근 금융 시장의 데이터 는 까다로운 신호 처리 문제를 제공하며 
그러한 데이터를 생성하는 금융 시스템의 복잡성이 높아서 시스템에서 발생하는 데이터는 비선형 및 
 [비정적 특성](https://www.investopedia.com/articles/trading/07/stationary.asp)을 모두 나타냅니다. 
 2003년 15%에 비해 2012년 시장 거래량의 85%가 자동 거래 시스템으로 전환되면서 데이터를 쉽게 이용할 수 있게 됨에 따라 
 최근 분석의 중요성이 커지고 있다.
* 기본분석에서 사용되는 정성적 정보에는 수익성, 배당금 지급 및 뉴스피드를 통해 제공되는 회사에 대한 뉴스 등 회사가 정기적으로 발행하는 보고서가 포함됩니다.
* Neural Network(인공신경망) 및 SVM(서포트벡터머신)을 포함한 고급 데이터 기반 모델은 기본분석방법과 함께 사용되어왔습니다.
* 위와 관련하여 흥미로운점은 뉴스피드 및 기탙 텍스트 정보 소스를 자동으로 수집하고 관련 정보를 추출할 수 잇는 자연어 처리 및 이해에 있습니다.
* 텍스트에서 추출한 정보는 여러 분야에서 성공적으로 적용되어왔는데 
 [IMB의 Watson](https://en.wikipedia.org/wiki/Watson_\(computer)), 
 [유전자 조절 네트워크 구축](https://link.springer.com/article/10.1007%2Fs11265-007-0148-4)이 이러한 예시입니다.
* 이러한 배경을 바탕으로 주식시장의 종가예측에 중점을 두고 주식시장과 관련된 뉴스 데이터에 텍스트 마이닝을 적용하는 [사례](https://people.kth.se/~gyozo/docs/financial-prediction.pdf)가 있었는데 본 논문과 마찬가지로 가격방향예측을 위해 Naïve Bayes분류자를 사용합니다. 입력데이터로는 뉴스피드 뿐만아니라 StockTwits , Google Trends , Wikipedia page views 을 사용하였습니다.   
* 위 사례에 대해 간략히 설명하자면 방대한 양의 트윗을 이용해 6개의 감정을 도출한 뒤  [Dow Jones Index](https://www.sciencedirect.com/topics/economics-econometrics-and-finance/stock-index)(다우존 지수)의 이동방향을 예측하는 데 사용되었습니다.
* 해당 논문에서는 뉴스에서 파생된 정보가 자산가치(주식가격)나 이동방향보다 시장 변동성의 2차 특성에 더 큰 영향을 미칠 것이라는 가설을 탐구합니다.  
  * 해당 논문에서는 자연어의 Feature Reduction에 효과적인 LDA모델을 구성하고 분류는 단순한 가정에서도 잘 작동하는 Naïve Bayes알고리즘을 사용하여 수행합니다.
  * 해당 논문에서 제시하는 방법은 기존의 시계열 모델과 통합하는 것과는 대조적으로 뉴스에서 파생된 정보만으로 인한 자산 가격 및 변동성 변화에 대한 예측을 하는 데 있습니다.


## 2. Data and inference algorithms

### 2.1 Financial data

 * 해당 논문에서는 입력데이터로 J.P Morgan 과 Goldman Sachs 두개의 주식과 NASDAQ 과 Dow Jonew 두개의 주가지수를 선택했습니다. 
 * 경험적으로 경험적으로 관찰한 바에 따르면 평균적으로 뉴스 발표와 시장에 대한 영향 사이에는 20분의 [지연](https://www.sciencedirect.com/topics/economics-econometrics-and-finance/lag-model)이 있습니다.
 * 2011년 9월 9일부터 2012년 9월 7일까지 위의 데이터를 ['The Bonnot Gang'](http://thebonnotgang.com)에서 가져왔습니다. 

### 2.2 News data
* 텍스트 데이터의 경우 , 2011년 9월에서 2012년 0월까지 로이터 미국 뉴스 아카이브의 일부를 사용했습니다. 
 로이터를 선택한 이유는 미국 시장 변동성을 고려할 때 
 미국 시장에 높은 변동성과 낮은 변동성이 모두 포함되어 있기 때문에 선택했습니다.
 * 로이터를 뉴스 매체로 사용하는 또 다른 장점은 매일 아카이브 페이지가 데이터를 수집하기 쉽다는 점 때문입니다.

### 2.3 Volatility estimation

* 변동성은 각 구간에 대한 로그 수익률의 분산으로 추정되었으며 , 1시간마다 변경되었습니다. 

<img src="/images/mason/2018-12-17/picture_1.png" width="200px">

*  Si는 i 시점에서의 총 자산 , ri는 i 시점에서의 로그수익률


### 2.4 Machine learning

* LDA(Latent Dirichlet Allocation) 와 Naïve Bayes classification가 연구에서 사용된 기계학습 모델입니다.
* LDA에 대해 간략하게 설명하자면 일련의 문서를 축소하는 방법으로 기본 주제로 분류할 수 있는 모델입니다. 
주제는 단어들의 집합으로 , 각 주제는 문서에서 나타날 확률을 가집니다. 
LDA는 문서를 효과적으로 줄이고 이해할 수 있는 주제를 만들지만 단점은 상대적으로 계산이 복잡해서 시간비용이 많이 드는 알고리즘입니다.
* 해당 주제 모델링은 문서집합에서의 학습을 통해 생성할 수 있는 주제를 추론합니다.
 
### 2.4.2. Naïve Bayes classification

 * Naive Bayes는 패턴 분류에 사용할 수 있는 다양한 학습 알고리즘 중 하나입니다. 
 텍스트 분류에서의 인기는 단순함 때문입니다. 특히 , 다변수 분류 설정에서 Naive Bayes는 각 기능을 다른 기능과 독립적으로 모델링합니다. 
 
## 3. Text processing

 * Python의 'urllib' 와 'requests'패키지를 이용해 뉴스데이터를 가져온다음 , 'BequtifulSoup'패키지를 이융해 HTML을 분석하고 정보를 추출합니다.
 * 재정용어 단어들의 집합을 구성한 뒤 해당 내용에 포함되지 않는 기사를 사용하지 않습니다.
 * 직접 확인하면서 관련성이 낮은 뉴스 카테고리가 추출되지 않을 때까지 필터링 키워드를 추가하여 남겨두었습니다. 
 * 추출된 텍스트는 일반적으로 사용하는 자연어 처리의 과정을 거쳤습니다.
  * [Stop-words](https://www.geeksforgeeks.org/removing-stop-words-nltk-python/)(불용어) , 짧은단어(length < 3) 그리고 웹페이지 주소를 제거했습니다 .  

### 3.1 Topic modeling

 * 기사 텍스트의 의미를 모델링하기 위해 LDA를 사용하여 기사 본문과 제목을 별도의 주제 모델로 사용하고 토픽 분포를 단일 피쳐 벡터로 결합하는 단계를 포함한다.
 * LDA 구현은 Python의 "gensim'패키지를 사용하였습니다. 
 * 우리는 LDA 모델에대한 파라미터를 선택했습니다 . 
  * Topics-body = 100 , Topics-title = 20 , LDAPasses-body = 3 , LDAPasses-title = 20  , LDAChunks-body = 2000 , LDAChunks-title =200
  * Hierarchical Dirichlet Process(HDP) LDA와 유사한 비 매개 변수식 주제 모델은 주제 수를 초기화하는 데 사용되었습니다.

### 3.2  Naïve Bayes prediction model

 * 토픽모델로부터 , 각 60분 예측 간격에 대한 사이즈 120(100 개의 본문 토픽, 20개의 제목 토픽)의 토픽리스트를 구성했습니다. 구간에 대한 특징 벡터는 해당 구간 내에서 각 주제가 기사에 등장하는 횟수를 나타내는 벡터입니다.일부 주제는 두번 이상 나타날 수 있으며 일부 주제는 전혀 표시되지 않을 수 있습니다.
 * 목표 벡터는 시장 데이터의 이진 방향 레이블이며 각 특징 벡터와 짝을 지어서 구성됩니다. 이렇게 구성된 벡터는 60분의 기간에 대한 방향레이블을 예측할 때 사용됩니다.
  * 예를들어 9:00 AM - 10:00 AM의 뉴스를 사용하여 10:00 AM - 11:00 AM volatility의 이동방향(증가 / 감소)을 예측합니다.
 * 위 방법을 이용해 다중 나이브 베이즈 모델을 학습시켰습니다. 6개월을 학습데이터로 사용하고 나머지 6개월을 테스트데이터로 사용하였습니다. 이 사이즈를 증가시키자 정확도가 확연하게 떨어졌는데 이는 금융 시장의 비 안정성에 기인했다고 생각합니다.
 * 또한 불규칙한 패턴을 가지는 경향이 있는 매일 주식시장 개장 이후 1시간 및 폐장 이전 30분은 사용하지 않았습니다.
 * 이러한 텍스트를 벡터료 표헌할때 발생하는 문제 중 하나인 용어 / 주제가 관찰되지 않는 간격 동안 확률이 0으로 설정되는 경우인데 이것은 [Lidstone smoothing parameter](https://en.wikipedia.org/wiki/Additive_smoothing#Statistical_language_modelling)에 의해 처리되며 파라미터(α)는 0.01로 설정됩니다.
 
### 3.3  Dealing with non-stationary data

 * 금융정보는 비정적이라고 알려져있습니다. 즉 짧은 기간에 우리가 매핑시킨 뉴스와 시장의 움직임이 맞지 않을 수 있습니다. 이런 이유로 우리는 예측시점에서보다 더 먼 시점의 데이터가 가까운 시점의 데이터보다 상대적으로 덜 중요할 것이라고 추정할 수 있습니다.
 * 이러한 가정을 적용하기 위해 감쇠함수를 구현하였고 60분의 뉴스를 6개의 간격으로 나누었고, 
 이어 각 간격에 걸쳐 감소하는 방식으로 가중치를 산정하였습니다.

![](/images/mason/2018-12-17/picture_2.png)

* i=0인 경우는 예측 기간에 가까운 지점 i = 5 인 경우는 예측 시점에서 가장 먼 지점에 있는 경우
* 가중치는 간격에 걸쳐 주제의 수에 가중치 w를 곱하여 주제모델에 적용하였고 [ad-hoc testing](http://softwaretestingfundamentals.com/ad-hoc-testing/)을 통해 개선되는것을 확인했습니다.
 
### 3.4 Bigram model

 * 기존 모델과 또 다른 개선점은 Bigram을 사용하는 것에 있습니다. Unigram보다 의미 정보를 더 잘 포착할 수 있다고 생각합니다
 * 예를들어, Unigram 모델의 경우 money와 market의 관련성이 사라지지만 Bigram의 경우는 money-market 같이 돈과 시장이 관련이 있는것을 포착할 수 있습니다.
 
### 3.5 Chi-squared feature reduction

 * 위 방식으로 구성된 피처세트는 매우 높은 차원인데 차원을 줄이기 위해 [Chi-squared feature reduction](http://koonja.co.kr/upload/goods/pdf/ED97A974AB676AEEA2AF685FD237037920120821132054.pdf)을 사용하였습니다. 높은 차원을 다루는 방법중에 [PCA](https://www.sciencedirect.com/topics/economics-econometrics-and-finance/principal-component-analysis)(주성분분석)이 많이 사용되지만 실험적으로 비교할 때 Chi-squared feature selection방법이 적은 분산의 정확도 결과를 산출했기 때문에 Chi-squared feature rediction 방법을 선택하였습니다
 
### 3.6 Luhn's cut-off

 * 자연어에서 가장 빈도가 높은 용어는 세 번째로 빈번하게 등장하는 단어의 4배이며 두번째의 두배입니다. 학습에서 자주 사용되는 단어의 우세로 인한 어려움과 훈련 및 시험 세트에만 나타나는 희귀한 단어로 인한 어려움을 피하기 위해 [Luhn cut-offs](https://ko.wikipedia.org/wiki/%EC%A7%80%ED%94%84%EC%9D%98_%EB%B2%95%EC%B9%99)라고 알려진 사용 빈도 테이블에서 상한 및 하한을 결정합니다. 그 결과 중요한 단어는 Cut - off에 속해있지 않은 단어보다 더 큰 판별력을 갖습니다.
 
## 4. Technical analysis model

### 4.1 Rationale

 * 예측을 위한 입력으로 뉴스만을 사용 모델만 생성하는 것이 아닌 시계열에서 파생된 변수를 사용하여 병렬모델을 구성했습니다. 
 이 모델은 변동성의 여부를 평가하는 벤치마크입니다. 
 
### 4.2 Implementation

 * 시계열 데이터는 매 시간 12개의 데이터 포인트([OHLC](https://en.wikipedia.org/wiki/Open-high-low-close_chart))를 
 생성하였습니다. 기술 지표로는 총 9개를 사용하였고 OHLC는 기술적 지표는 아니지만 OHLC의 평균 및 변동성도 함께 입력하여 
 시간당 180개의 변수를 생성했습니다.
 * 최종 특징 벡터는 뉴스 기반의 예측 모델과 동일한 방식으로 변동성의 (상승 / 하강) 및 가까운 시점을 기준으로 
 각 입력에 대해 t 에서 t -55 까지의 지연된 입력을 나타냅니다.
 Python의 TA-Lib패키지를 사용하였고 , 입력벡터는 훈련 빛 학습 알고리즘이 수행되기 전에 
 평균 및 단위 분산이 0인 가우시안 분포에 맞게 조정한 뒤 수행하였습니다.
 
## 5. Benchmarks and measures
 * 뉴스 기반 모델 (모델N)의 성능평가와 뉴스정보와 시장 움직임 간의 관계를 테스트하기 위해 다양한 방법을 사용하였습니다. 

### 5.1 Single direction (SD)
 * 짦은 기간동안 시장 움직임은 단조롭기 때문에 항상 위 또는 아래로 이동방향을 예측하는 것이 유용한 벤치마크입니다.
 
### 5.2 Random walk (RW)
 * [랜덤워크](https://ko.wikipedia.org/wiki/%EB%AC%B4%EC%9E%91%EC%9C%84_%ED%96%89%EB%B3%B4)시간에 따른 편차의 평균이 0이지만 분산은 시간에 비례하여 증가하게 된다. 따라서, 앞뒤로 움직일 확률이 동일하다고 해도 시간이 흐름에 따라 평균에서 점차 벗어나는 경향을 보인다.

### 5.3 Technical analysis model (ModelTA)
 * ModelTA는 시계열 데이터만을 기반으로 예측을 형성합니다.

### 5.4. Classifier performances
 * 분류 성능을 정량화하기 위해 accuracy, recall, precision, F1 score , Matthews correlation coefficient와 같은 정보검색에서 많이 쓰이는 여러가지 방법을 사용했습니다.
 * 시장 데이터에서 움직임이 매우 불균형을 일으킬 수 있기 때문에 성능측정항목으로 다양한 측정 값을 사용합니다.
 
 ![](/images/mason/2018-12-17/picture_3.png)
 
## 6. Results

 * Model N : 뉴스데이터를 기반으로 생성한 모델
 * Model TA : 주식시장의 시계열데이터를 기반으로 생성한 모델
 
 * Model N을 F1평가를 이용해 측정해본 결과 변동성 변화에 대한 예측은 직접적인 시장 가격 예측보다 훨씬 나은 성능을 보여주었습니다.
 * Model N의 모든 테스트구간에서의 정확도는 변동성 :  55.6% , 종가 : 49.4% 입니다.
 
 ## 7. Conclusiion and discussion
 
 * 해당 논문에서의 텍스트 뉴스 소스에서 추출한 정보를 시장 변동성의 방향성 변화를 예측하는 데 사용할 수 있다는 결과를 확인했습니다.
 
 * 생성도니 모델의 결과는 뉴스 출처에서 파생 된 정보에만 근거하고 다른 입력은 사용되지 않는 점에 유의해야합니다. 따라서 정확도 결과는 시계열을 기반으로 예측하는 모델과 직접 비교하기는 어렵지만 뉴스에서 추출한 정보만으로 시장변동성의 이동방향을 예측할만큼 강력한 신호 수준을 포함하고 있다는 것은 눈여겨 볼 필요가 있습니다.
 
