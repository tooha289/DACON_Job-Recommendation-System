import numpy as np

def RecallPrecision_ATk(test_data, r, k):
    right_pred = r[:, :k].sum(1)
    precis_n = k
    recall_n = np.array([len(test_data[i]) for i in range(len(test_data))])
    recall = np.sum(right_pred/recall_n)
    precis = np.sum(right_pred)/precis_n
    return {'recall': recall, 'precision': precis}

def NDCGatK_r(test_data,r,k):
    assert len(r) == len(test_data)
    pred_data = r[:, :k]

    test_matrix = np.zeros((len(pred_data), k))
    for i, items in enumerate(test_data):
        length = k if k <= len(items) else len(items)
        test_matrix[i, :length] = 1
    max_r = test_matrix
    idcg = np.sum(max_r * 1./np.log2(np.arange(2, k + 2)), axis=1)
    dcg = pred_data*(1./np.log2(np.arange(2, k + 2)))
    dcg = np.sum(dcg, axis=1)
    idcg[idcg == 0.] = 1.
    ndcg = dcg/idcg
    ndcg[np.isnan(ndcg)] = 0.
    return np.sum(ndcg)

def read_data(data):
    Nu = 8482
    Ni = 6695
    
    f = open('../data/{}/train.txt'.format(data), 'r')
    lines = f.readlines()
    lines = [l.strip('\n\r') for l in lines]
    lines = [l.split(' ') for l in lines]
    train = [l[1:] for l in lines]
    for i in range(len(train)):
        if train[i] == [""]:
            train[i] = []
    train = [[int(x) for x in i] for i in train]

    f = open('../data/{}/test.txt'.format(data), 'r')
    lines = f.readlines()
    lines = [l.strip('\n\r') for l in lines]
    lines = [l.split(' ') for l in lines]
    test = [l[1:] for l in lines]
    for i in range(len(test)):
        if test[i] == [""]:
            test[i] = []
    test = [[int(x) for x in i] for i in test]
    
    M_train = np.zeros([Nu, Ni]) # 'M_train' 변수 초기화 (0으로 초기화)
    # [M_train]: 크기가 Nu x Ni인 이진 행렬로 '사용자-항목 간의 상호작용'을 나타낸다.
    # [0]: 상호작용 O
    # [1]: 상호작용 X

    for u in range(Nu):
        if len(train[u]) != 0:
        # 현재 사용자-항목 간 상호작용 여부 확인(상호작용 X 사용자는 건너뛰고 다음 사용자로 이동)
            for i in range(len(train[u])):
            # 현재 사용자가 상호작용한 항목의 목록에 대한 반복문(i: 현재 항목)
                M_train[u, int(train[u][i])] = 1
                # 사용자-항목 간 상호작용을 1로 표시(u: 현재 사용자, train[u][i]: 항목)
                
    return Nu, Ni, M_train, train, test
    # [Nu]: 사용자 수
    # [Ni]: 항목 수
    # [M_train]: 사용자-항목 간 상호 작용을 나타내는 이진 행렬(1=상호작용 O, 0=상호작용 X) 
    # [train]: '훈련 세트'에서 상호작용한 항목을 나타낸다.
    # [test]: '테스트 세트'에서 상호작용한 항목을 나타낸다.

def scores(train, test, rec, Nu, Ni, K, data_set): # [Nu]:사용자 수, [Ni]:항목 수, [0]:상호작용O, [1]:상호작용X
    ndcgK = 0.0 # ndcgK 변수 초기화
    recK = 0.0 # recK 변수 초기화
    precK = 0.0 # precK 변수 초기화
    R = [] # 빈 리스트 R 설정/초기화 (사용자별로 추천된 항목을 저장하는 목적으로 사용)

    pred_prob_list = [] # ★recommend_prob 추가용★
    pred_idx_list = [] # ★recommend_idx 추가용★

    for u in range(Nu):
        pred = rec[u,:] # rec 배열의 'u'번째 행(해당 user에 대한 추천 점수)
        true = np.zeros([Ni]) # 항목을 0으로 초기화된 배열로 설정(user가 test 데이터에서 상호작용한 항목을 나타내기 위해 사용)
        true[test[u]] = 1 # test 배열에서 'u'번째 user가 상호작용한 항목의 인덱스를 추출하고 해당 위치를 1로 설정하여 true 배열을 업데이트
        # pred = np.delete(pred, train[u]) # 'pred'에서 train[u]에 있는 항목들 삭제 (기존 훈련데이터에 있던 항목 → '추천'에서 제외(중복 방지))
        # true = np.delete(true, train[u]) # 'true'에서 train[u]에 있는 항목들 삭제 (기존 훈련데이터에 있던 항목 → '실제 상호작용'에서 제외(중복 방지))
        pred[train[u]] = 1e-12 # ★test code 추가★ (위 주석 처리한 코드 2줄을 해당 코드로 대체(개수 변화 방지))

        #############################################################################
        pred_prob = (pred - np.min(pred)) / (np.max(pred) - np.min(pred)) # score 계산 코드

        idx = np.flip(pred_prob.argsort()) # ★recommend_prob를 위한 idx★
        pred_prob_list.append(pred_prob[idx[:K]])

        idx = np.flip(pred.argsort()) # ★recommend_idx를 위한 idx★
        pred_idx_list.append(idx[:K])
        #############################################################################

        # [argsort]: 배열을 정렬했을 때의 각 요소의 인덱스를 반환(오름차순 기준 원래 인덱스(정렬X))
        # [np.flip]: 배열의 요소를 역순으로 뒤집는 함수
        # 'pred' 배열에 대한 인덱스를 내림차순 정렬(추천된 항목을 '추천 점수 순'으로 정렬하기 위함)

        R.append(list(true[idx[:K]])) # 'true' 배열에서 상위 'K'개 항목을 추천 목록 'R'에 추가(이 추천 목록은 user별로 저장된다.)
        # print(R)

        scor = RecallPrecision_ATk(test_data = [test[u]], r = np.array([R[u]]), k = K)
        # Recall과 Precision을 계산 (이 함수는 사용자의 테스트 데이터와 추천 목록을 기반으로 Recall과 Precision을 계산)
        precK += np.nan_to_num(scor["precision"]/Nu)
        # Precision을 현재 사용자에 대한 결과에 누적
        recK += np.nan_to_num(scor["recall"]/Nu)
        # Recall을 현재 사용자에 대한 결과에 누적
        ndcgK += np.nan_to_num(NDCGatK_r(test_data = [test[u]], r = np.array([R[u]]), k = K)/Nu)
        # NDCG 값을 현재 사용자에 대한 결과에 누적
    
    SAVE_PATH = '../results/'
    # [recommend_idx 파일 저장용]
    with open(f'{SAVE_PATH}pred_idx_{data_set}.txt', 'w') as file:
        for u, rec_idx in enumerate(pred_idx_list):
            file.write(f"{u} {' '.join(map(str, rec_idx))}\n")
    
    # [recommend_prob 파일 저장용]
    with open(f'{SAVE_PATH}pred_prob_{data_set}.txt', 'w') as file:
        for u, rec_prob in enumerate(pred_prob_list):
            file.write(f"{u} {' '.join(map(str, rec_prob))}\n")

    return precK, recK, ndcgK