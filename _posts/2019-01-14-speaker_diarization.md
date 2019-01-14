---
layout: single
title:  "About Speaker Diarization"
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
toc: true
toc_label: "목차"
toc_icon: "cog"

author: daydrill
---




1. INTRO

	* 라디오에서 토론 프로그램을 듣는 상황을 생각해보자. 두명 이상의 화자가 서로 어떤 안건에 대해 맹렬히 찬반을 논하고 있다. 라디오를 통해 우리는 하나의 오디오 스트림이 있을 뿐이지만, 그 안의 여러명의 대화를 우리는 끊어서 이어 붙이며 누가 어떤 말을 하고 있는지 이해 할 수 있다. 
	* 이름도 생소한 speaker diarization은 컴퓨터가 이러한 과정을 하는 것을 의미한다. 
	* 쉽게 말하면 음성 혹은 비디오 파일에서 “who spoke when?”에 대한 해답을 찾는 문제이다.
	* 이를 위해서 발화자가 있다고 판별된 오디오 스트림과 각 구간에서 비지도적인 식별이 필요하다.
	* 이번 포스트에서는 <Speaker Diarization: A Review of Recent Research>라는 2012년 공개된 논문을 통해 SD가 무엇인지 간단하게 살펴보고, 이를 위해 어떤 알고리즘이 필요한지 알아보자.



2. SPEAKER DIARIZATION 이란?
  * 1990년도 후반과 2000년도 초반, SD가 연구되던 초기에는 전화 음성과 방송 뉴스가 주요 연구 주제였고, 전세계에 송출되는 모든 TV와 라디오 프로그램의 자동 주석(annotation)을 목표로 했다.
	* 그러다 2002년부터 M4, IM2, AMI 등의 여러 프로젝트가 생겨나면서 미팅에서의 SD가 주목을 받기 시작했다.
	* 이러한 연구들과 멀티모달 기술 개발은 자동으로 미팅 내용을 추출하거나 미팅 참여자의 정보를 뽑아내거나 아카이빙 목적으로 human-to-human 커뮤니케이션의 향상에 헌신했다. 




4. Main Approach
	* SD에 대한 접근 방법은 여러가지가 있지만, 그 중에서 많이 사용되는 방식으로 Top-Down 접근 방식과 Bottom-UP 방식이 있다. 
	* 두 방식 모두 반복적인 연산으로 최적의 클러스터 갯수로 수렴하는 것을 목표로 한다.
	* 이를 위해 일반적으로 HMM(Hidden Markov Models)과 GMM(Gaussain Mixture Models)를 기반의 알고리즘을 이용한다.
	* 각각 접근 방식을 알아보자!
    * Bottom-Up Approach
  		* SD를 위해 가장 많이 사용되는 알고리즘이고, AHC(Agglomerative hierarchical clustering)으로 말하기도 한다.
     	* 이 방식은 클러스터의 수 혹은 모델을  연속적으로 병합
    * Top-Down Approach
		  * 아주 적은(일반적으로 하나) 클러스터를 점점 쪼개는 방식


4. 주요 알고리즘
	* 일반적인 SD는 다음 그림과 같은 순서대로 수행하며, 이를 위해 필요한 몇가지 알고리즘의 기본 개념에 대해 알아보자.
	* Acoustic Beamforming
    * 미팅에서 SD를 가정해보자. 여러개의 마이크를 통해 음향 정보를 가져온다. 이렇게 멀티채널 일 경우에서 SD에서의 피쳐로 사용하기를 원한다. 
		* 마이크는 그 종류마다 다른 특징을 가지고 있다. 
		* 
	* Speech Activity Detection

		* SAD(Speech Activity Detection)는 거의 모든 음성 처리(음성코딩, 음성향상, 음성인식 등..)의 기초적인 타스크이다.
		* 오디오 스트림의 각 구간 음성(speech) 구간인지 비음성(non-speech) 구간인지 판별하는 과정이다.
		* 비음성 구간의 경우에는 silence 뿐만 아니라 책 넘기는 소리, 바람소리, 노크소리 같은 환경 소음(ambient noise), 그리고 숨소리, 기침소리, 웃음소리 같은 비어휘 소음(non-lexical noise)까지도 포함한다.
		* SAD는 두가지 이유로 SD의 퍼포먼스를 좌우한다.
		* 첫번째로, SD의 성능 측정 방식인 DER(diarization error rate)이 SAD의 거짓 경보(false alarm)와 틀린 speaker error rate를 모두 포함하여 산정하기 때문이다.
		* 두번째로 비음성 구간이 앞서 살펴봤던 SD의 비지도적인 학습 과정을 방해 할 수 있다다는 것이다.
	* Segmentation

		* speaker segmentation은 SD 과정의 핵심이며, 화자에 대하여 homogeneous 하도록 오디오 스트림을 잘게 구분하는 과정이다.
		* 다시 말해 각 구간은 한명의 화자의 발화만 있어야 한다.
		* 세그멘테이션에 대한 고전적인 접근 방식은 두개의 슬라이딩 




[참고자료]
	* Speaker Diarization: A Review of Recent Research : https://ieeexplore.ieee.org/document/6135543

