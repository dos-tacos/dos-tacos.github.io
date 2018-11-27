# Guideline for 'Delicious tacos, Savory machine learning'
- 블로그 url: https://dos-tacos.github.io/
- 이하의 설명들은 `$HOME = dos-tacos.github.io의 root 위치`를 전제로 함
- 본 블로그는 Jekyll의 [minimal-mistakes](https://mmistakes.github.io/minimal-mistakes/) 테마를 사용
- 공식 문서 [Quick start guide](https://mmistakes.github.io/minimal-mistakes/docs/quick-start-guide/)


## Posts
### 기본 작성 요령
- `${HOME}/_posts` 안에 `md` 확장자로 작성 ([markdown 문법 참고](https://github.com/adam-p/markdown-here/wiki/Markdown-Cheatsheet))
- 각 포스트의 제목은 `yyyy-mm-dd-title.md` 형태로 작성(띄어쓰기 없음. 공백 있을 시 하이픈으로 대체)
  - 이 문서 제목이 해당 문서로의 url이 됨
  
### 문서 내 메타데이터
- 각 포스트의 상단에는 아래와 같은 메타데이터 작성
```markdown

---
layout: single
title:  "The First Post"
header:
  teaser: "images/unsplash-gallery-image-2-th.jpg"
categories: 
  - update
tags:
  - test
  - update
author: Lynn Hong
---

The first blog post.
```
- layout: post의 모양
- title: post가 목록에 노출될 때의 문서 제목
- header: post 상단의 header가 존재하는 layout일 경우의 header 이미지 파일
- categories: 해당 post에 부여할 주제 카테고리. **하나만 등록하는 것을 권장**
- tags: 해당 post의 키워드. 띄어쓰기 가능, 여러 개 등록 가능
- author: post 좌측에 보여질 작성자 정보(사전에 `authors.yml`에 프로필 넣어 두어야 함. 이하에서 설명)


## Images
- 이미지가 저장될 위치는 `${HOME}/images/{본인 이름 폴더}/{날짜}` 권장
  - 여기서 `${HOME}/images/{본인 이름 폴더}`까지는 필수적으로 지켜야 하며, 해당 폴더 내에서는 정리 방식 자유
  - 본인 이름 폴더는 본인이 직접 생성
  - 한 포스트에 첨부된 이미지가 개수가 많을 것으로 예상되므로 권장이긴 하지만 날짜 기준으로 분류해 놓으면 좋을 듯함
- post 내에서 이미지 파일 경로 넣을 때는 `"images/{본인 이름 폴더}..."`로 기재
  - 이 때 이미지 파일의 이름은 대소문자를 구분하므로 확장자가 혹시 대문자가 아닌지 잘 확인
- 일반 에디터를 사용하는 것보다 github pages에서 이미지 다루기가 조금 불편하긴 하지만 어쩔 수 없습니다 ㅜㅜ
  

## Author information
### author 정보 작성 요령
- 본인이 작성한 글을 눌렀을 떄 좌측에 프로필이 나타나게 하려면 `${HOME}/_data/authors.yml` 파일을 수정
- 프로필 사진의 위치는 `${HOME}/images/common`에 업로드
  - 사진은 1:1 비율(정사각형)에 최적화
  - 모서리가 라운드된 원형으로 보이므로 약간 잘리는 것 감안하고 업로드

### author 정보 메타데이터
- 아래 예시를 기초로 작성
```markdown
Lynn Hong:
  name        : "Lynn Hong"
  #uri         : "http://thewhip.com"
  bio         : "Instead of looking at things, look between things."
  avatar      : "/images/common/hong_su_lyn.jpg"
  links:
    - label: "Email"
      icon: "fas fa-fw fa-envelope-square"
      url: "mailto:lynnn.hong@gmail.com"
    - label: "Facebook"
      icon: "fab fa-fw fa-facebook-square"
      url: "https://web.facebook.com/sulynhong"
```
- name: 본인 이름
- uri: 대표 링크. 사진을 누르면 이동하는 url로 추정
- bio: 사진 밑에 나오는 한 마디 말
- avatar: 개인 프로필 사진
- links: `bio` 밑에 나오는 외부 링크로, 개수 제한 없이 커스터마이징 가능
  - label: 아이콘 옆에 나오는 해당 링크의 메뉴명
  - icon: 아이콘 font. [font awesome](https://fontawesome.com/icons)에서 검색하여 사용
  - url: 연결하고 싶은 url
- [theme official document(authors)](https://mmistakes.github.io/minimal-mistakes/docs/authors/)


## Blog update
- 일단은 담당자가 @lynn-hong 입니다
- 현재 branch를 `master`만 만들어 두었으므로 위에 적은 폴더나 문서 외에 수정하면 충돌이 발생할 수 있습니다
- 업데이트는 지속적으로 하고 있으니 **원하는 기능이나 버그 있을 경우 issue에 남겨주세요**
- 직접 수정하고 싶을 경우 branch 하나 파서 수정 후 `master`로 pull request 보내주세요~
