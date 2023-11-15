- [Background](#background)
- [Theme](#theme)
- [Description](#description)
- [Host / Organizer](#host--organizer)
- [Dataset Information](#dataset-information)
- [Reference and Citation](#reference-and-citation)
  - [BSPM \[link\]](#bspm-link)

# 배경
국민대학교 경영대학원 AI빅데이터/디지털마케팅전공과 경영대학에서 ‘제1회 국민대학교 AI빅데이터 분석 경진대회’를 개최합니다.

이번 대회에서는 Total HR Service를 제공하는 (주)스카우트의 후원을 받아 유연한 노동시장으로의 변화 흐름에 맞추어,

구직자 입장에서는 자신의 이력과 경력에 맞춤화된 채용 공고를 추천받을 수 있고 구인기업 입장에서는 공고에 적합한 인재를 미리 선별하는 도구로 활용할 수 있도록 채용공고 추천 알고리즘 개발을 제안합니다.

이력서 등 구직자 관련 데이터와 채용 공고 관련 데이터, 그리고 지원 히스토리 데이터를 활용하여 구직자에게 맞춤화된 채용 공고를 자동으로 추천할 수 있는 알고리즘을 개발함에 따라

지원자는 적성에 맞는 채용 공고에 지원하여 직무 만족도를 높이고 구인기업은 직종에 맞는 핵심 인재를 선발할 수 있으리라 기대할 수 있습니다.

# 주제
이력서 맞춤형 채용 공고 추천 AI 모델 개발

# 설명
이력서, 채용 공고 및 지원 히스토리 데이터를 활용하여 구직자에게 맞춤화된 채용 공고를 자동으로 추천할 수 있는 추천시스템 알고리즘 개발

# 주최 / 주관
* 주최 : 국민대학교 경영대학원, LINC+ 사업단
* 주관 : 국민대학교 경영대학, 국민대학교 경영대학원 AI빅데이터전공/디지털마케팅전공
* 운영 : 데이콘
* 후원 : (주)스카우트

# 데이터 설명

- `apply_train.csv` [파일]
  - 이력서가 채용 공고에 실제 지원한 관계 목록 (히스토리)
  
- 이력서 관련 데이터 [파일]
  - `resume.csv`
  - `resume_certificate.csv`
  - `resume_education.csv`
  - `resume_language.csv`

- 채용공고 관련 데이터 [파일]
  - `recruitment.csv`
  - `company.csv`

- `sample_submission.csv` [파일] -  제출 양식
  - `resume_seq`: 추천을 진행할 이력서 고유 번호
  - `recruitment_seq`: 이력서에 대해 추천한 채용 공고 고유 번호
  - resume.csv에 존재하는 모든 resume_seq에 대해서 5개의 채용 공고를 추천해야 합니다.
  - 해당 이력서에서 실제 지원이 이루어졌던 채용 공고는 추천하지 않습니다.

※ 상세 데이터 명세는 [링크](https://dacon.io/competitions/official/236170/talkboard/409868?page=1&dtype=recent)를 반드시 참고해주세요.

# 개발 환경

# 라이브러리 환경

# 실행
## 제출 결과 재현

## 모델 별 실행

# 참조 및 인용
## BSPM [[link]](https://github.com/jeongwhanchoi/BSPM)
```
@inproceedings{choi2023bspm,
  title={Blurring-Sharpening Process Models for Collaborative Filtering},
  author={Choi, Jeongwhan and Hong, Seoyoung and Park, Noseong and Cho, Sung-Bae},
  booktitle={Proceedings of the ACM Conference on Research and Development in Information Retrieval (SIGIR)},
  year={2023}
}
```
## LT-OCF [[link]](https://github.com/jeongwhanchoi/LT-OCF/tree/main)
```
@inproceedings{choi2021ltocf,
  title={LT-OCF: Learnable-Time ODE-based Collaborative Filtering},
  author={Choi, Jeongwhan and Jeon, Jinsung and Park, Noseong},
  booktitle={Proceedings of the 30th ACM International Conference on Information and Knowledge Management},
  year={2021},
  organization={ACM}
}
```
## SSCF [[link]](https://github.com/giamba95/SaplingSimilarity/tree/main)
This repository contains the implementation for our paper:

Sapling Similarity: a performing and interpretable memory-based tool for recommendation [https://arxiv.org/abs/2210.07039](https://arxiv.org/abs/2210.07039)
