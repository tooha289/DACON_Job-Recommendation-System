- [배경](#배경)
- [주제](#주제)
- [설명](#설명)
- [주최 / 주관](#주최--주관)
- [데이터 설명](#데이터-설명)
- [개발 환경](#개발-환경)
- [라이브러리 환경](#라이브러리-환경)
- [실행](#실행)
  - [제출 결과 재현](#제출-결과-재현)
  - [모델 별 실행](#모델-별-실행)
    - [LT-OCF](#lt-ocf)
      - [Parameters](#parameters)
    - [BSPM](#bspm)
      - [Parameters](#parameters-1)
    - [SSCF](#sscf)
      - [Parameters](#parameters-2)
- [참조 및 인용](#참조-및-인용)
  - [BSPM \[link\]](#bspm-link)
  - [LT-OCF \[link\]](#lt-ocf-link)
  - [SSCF \[link\]](#sscf-link)

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
**Install python environment**

```
conda env create --file environment.yml
```

**Activate environment**

```
conda activate job
```
# 실행
## 제출 결과 재현
1. 

## 모델 별 실행
### LT-OCF
* 학습 가능한 시간 기반의 미분 방정식을 활용한 협업 필터링 방법
* 미분 방정식의 개념인 NODEs(Neural Ordinary Differential Equations)위에 레이어 조합과 함께 Linear GCN을 재설계

**In terminal**

`LT-OCF/code`위치를 현재 디렉토리로 설정합니다. 
```
python main.py --dataset="JOB" --model="ltocf" --solver="rk4" --adjoint=False --K=4 --learnable_time=False --dual_res=False --lr=1e-3 --lr_time=1e-3 --decay=1e-4 --topks="[20]" --tensorboard=1 --gpuid=0 --epochs=320 --layer=2 --recdim=360 --bpr_batch=2048 --pretrain=0
```

#### Parameters
`parse.py` 에서 더 많은 파라미터를 확인 할 수 있습니다.
* `gpuid` (default: 0):
  
  GPU를 사용하는 경우, 학습 및 예측에 사용할 GPU의 ID를 나타냅니다. 
* `dataset` (JOB):
  
  사용할 데이터셋을 선택하는 파라미터입니다. 주어진 옵션은 "JOB"만이 존재합니다.
* `model` (ltocf):
  
  사용할 모델을 선택하는 파라미터입니다. 여기서는 "ltocf"를 사용합니다.
* `solver` (euler, rk4, implicit_adams, dopri5):
  
  ODEs를 해결하기 위한 수치적인 방법을 지정하는 파라미터입니다. 주어진 옵션으로는 Explicit Euler Method (euler), Runge-Kutta Method (rk4), Implicit Adams Method (implicit_adams), Dormand-Prince Method (dopri5) 등이 있습니다.
* `adjoint` (False, True):

  Adjoint ODE Solver를 사용할지 여부를 결정하는 파라미터입니다. Adjoint Solver는 ODEs를 더 효율적으로 해결하는 데 도움이 될 수 있습니다.
* `K` (1, 2, 3, 4):
  
  ODEs를 푸는 시간 범위를 결정하는 파라미터로, K 값은 시간의 최대 범위를 나타냅니다.
* `learnable_time` (True, False):

  모델이 학습 가능한 시간을 사용할지 여부를 결정하는 파라미터입니다. 만약 True로 설정되면, 학습 중에 시간 변수가 조절 가능하며, False로 설정되면 고정된 시간이 사용됩니다.
* `dual_res` (False, True):
  
  이 파라미터가 True로 설정되면, 모델은 dual residual connections를 사용하여 ODEs를 계산합니다. 이는 Learnable-time Architecture에 대한 개선된 형태를 나타냅니다.
* `lr`:

  학습률을 나타내는 파라미터입니다.
* `lr_time`:

  시간에 대한 학습률을 나타내는 파라미터입니다.
* `decay`:

  가중치 감쇠 (weight decay)를 나타내는 파라미터입니다.
* `topks`:

  성능 측정에서 사용할 Top-K 값들을 나타내는 파라미터입니다.
* `epochs`:

  전체 학습 에폭 수를 나타내는 파라미터입니다.
* `layer`:

  뉴럴 네트워크의 레이어 수를 나타내는 파라미터입니다.
* `recdim`:

  임베딩 차원을 나타내는 파라미터입니다.
* `bpr_batch`:

  BPR 손실을 계산할 때 사용되는 배치 크기를 나타내는 파라미터입니다.
* `seed`(default: 2020):

  랜덤 시드 파라미터입니다.

### BSPM
* 학습 가능한 시간 기반의 미분 방정식을 활용한 협업 필터링 방법
* 미분 방정식의 개념인 NODEs(Neural Ordinary Differential Equations)위에 레이어 조합과 함께 Linear GCN을 재설계

**In terminal**

`BSPM/bspm`위치를 현재 디렉토리로 설정합니다. 
```
python main.py --dataset="JOB" --topks="[20]" --simple_model="bspm" --solver_shr="rk4" --K_s=1 --T_s=3.5 --final_sharpening=True --idl_beta=0.3 --factor_dim=960
```

#### Parameters
`parse.py` 에서 더 많은 파라미터를 확인 할 수 있습니다.
* `final_sharpening`:
  
  최종 Sharpening 단계에서의 합성 방법을 결정합니다.
  - True: Early Merge (EM) - Sharpening과 Blurring의 결과를 조합하여 최종 결과 생성
  - False: Late Merge (LM) - Sharpening과 Blurring의 결과를 따로 유지
* `solver_shr`:

  Sharpening ODE(Ordinary Differential Equation) 해법을 선택합니다.
  - euler: Euler method
  - rk4: Runge-Kutta 4th order method
  - K_s: Sharpening의 단계 수 (The number of sharpening steps)

* `T_s`:

  Sharpening ODE의 종료 시간 (The terminal time of the sharpening ODE)입니다.

* `t_point_combination`:

  모델에서 사용할 시간 포인트 조합 방법을 설정합니다.

  - True: 다양한 시간 포인트의 조합을 사용하여 모델링
  - False: 단일 시간 포인트만 사용하여 모델링
* `idl_beta`:

  IDL에서 사용되는 베타 값으로, blurring과 sharpening 간 상호 작용의 강도를 제어합니다. 높은 베타 값은 강한 상호 작용을 나타냅니다.

* `factor_dim`:

  잠재 요인의 차원으로, 모델이 학습하는 잠재적 특징의 수를 나타냅니다. 더 높은 잠재 요인 차원은 모델의 복잡성을 증가시킬 수 있지만, 과적합의 위험을 증가시킬 수도 있습니다.
* `seed`(default: 2020):

  랜덤 시드 파라미터입니다.

### SSCF
* 

**In terminal**

`SSCF/sscf`위치를 현재 디렉토리로 설정합니다. 
```
python main.py --dataset="JOB" --test="test" --gamma=0.2 --similarity="pearson"
```

#### Parameters
* `gamma`: 

사용자 기반 및 아이템 기반 접근법을 결합하는 데 사용되는 감마 매개 변수의 값입니다. (기본값: 0.5).

* `dataset`: 

사용 가능한 데이터셋을 선택하는 옵션입니다. 가능한 값은 [JOB].

* `similarity`: 

사용할 유사성 지표를 선택하는 옵션입니다. 가능한 값은 [common_neighbors, jaccard, adamic_adar, resource_allocation, cosine_similarity, sorensen, hub_depressed_index, hub_promoted_index, taxonomy_network, probabilistic_spreading, pearson, sapling_similarity] 중 하나 (기본값: 'sapling_similarity').

* `test`:

 모델을 적용할 데이터를 선택하는 옵션입니다. 가능한 값은 [validation, test] 중 하나 (기본값: 'test').


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
