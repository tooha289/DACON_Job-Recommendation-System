import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import utils
import similarities as sml
from parse import parse_args
import time
import csv

t1 = time.time() # 실행 시작 시간

args = parse_args() # 명령줄 인수를 구문 분석하는 함수를 호출(결과는 'args'에 저장)

# [폴더명별로 경로 설정하는 코드]
# base_path = '../data/split_params/'
# folder_list = [folder for folder in os.listdir(base_path) if os.path.isdir(os.path.join(base_path, folder))]

k_num = 20
similarity_list = [args.similarity]
gamma_list = [args.gamma]

with open('../results/sscf_recall_result.csv', mode='w', newline='') as file:
    field_names = ['gamma', 'similarity', 'recall@20']
    writer = csv.DictWriter(file, fieldnames=field_names)
    writer.writeheader()
    
    for similarity in similarity_list:
        for gamma in gamma_list:
            
            args.similarity = similarity
            args.gamma = gamma

            g = args.gamma # '감마' 값 설정
            data = args.dataset # dataset 설정(추천 시스템에 사용되는 dataset의 이름이나 식별자를 저장)
            model = args.similarity # 추천 시스템에 사용될 '유사성 계산의 유형'이나 방법을 저장
            test_data = args.test # 'test', 'validation' dataset 사용 여부 나타내는 변수 설정

            print(f"\n[gamma] '{g}', [similarity] '{similarity}'\n")
            
            print("reading data...")
            N_users, N_items, M, train, test = utils.read_data(data)
            M = M.astype(np.float32)
            # [N_users] 사용자 수[Nu]
            # [N_items] 아이템 수[Ni]
            # [M] 데이터 매트릭스(사용자-아이템 상호작용)

            # [사용자 간의 유사성 행렬]
            print("measuring similarity of users...")
            B = sml.similarity(M, model, 0)

            # [사용자 기반 추천 계산]
            print("measuring user-based recommendations...")
            rec_u = np.nan_to_num(np.dot(B,M).T/np.sum(abs(B), axis=1)).T
            # [rec_u] '각 사용자에 대한 아이템 평점 예측'을 나타내는 행렬

            # [아이템 간의 유사성 행렬]
            print("measuring similarity of items...")
            B = sml.similarity(M, model, 1)
            # ([M] Data Matrix, [model] 유사성 계산 방법, [1] 아이템 간의 유사성)

            # [아이템 기반 추천 계산]
            print("measuring item-based recommendations...")
            rec_i = np.nan_to_num(np.dot(M,B)/np.sum(abs(B), axis=0)) 

            # [최종 모델의 추천 계산]
            print("measuring final model recommendations...")
            rec = (1-g)*rec_u+g*rec_i # 최종 추천 모델(rec)
            # [g] rec_u와 rec_i의 조합 조정 매개변수(감마)
            # [g가 0에 근접] → '사용자 기반 추천' 영향력↑ 
            # [g가 1에 근접] → '아이템 기반 추천' 영향력↑ 

            # ['rec'(최종 추천 모델) 성능 측정]
            print("measuring performance of final model...")
            scores = utils.scores(train, test, rec, N_users, N_items, K=k_num)
            # [K]: 상위 K개 의미
            # [scores]: utils.scores()의 결과로 (Precision, Recall, NDCG)의 값들을 포함
            # [Reacll(재현율)]: 실제로 사용자가 선호하는 아이템 중 상위 K개 아이템에 포함되는 아이템의 비율

            print("\n[final model]")
            print("recall@20: {:.5f}\n".format(scores[1]))
            recall_20 = scores[1]
            writer.writerow({'gamma': gamma, 'similarity': similarity, 'recall@20': recall_20})
            print('-'*80)

t2 = time.time() # 실행 종료 시간
run_time = f'{t2-t1:.2f}'
print(f'Total Time: {run_time} seconds') # 실행에 걸린 총 시간

