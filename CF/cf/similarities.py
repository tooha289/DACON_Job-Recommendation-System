import numpy as np

# M: data matrix(행-사용자, 열-아이템, 셀-사용자와 아이템 간의 상호 작용을 나타내는 값)
# projection: 0, 1로 구분되는 프로젝션 방향

# [공통 이웃 유사도]
def common_neighbors(M, projection):
    if projection == 0:
        B = np.dot(M,M.T)
    else:
        B = np.dot(M.T,M)
    return B

# [Jaccard 유사도]
def jaccard(M, projection):
    if projection == 0:
        k = np.sum(M, axis = 1)
        coo = np.dot(M,M.T)
        B = np.nan_to_num(coo/(np.subtract.outer(k, -k)-coo))
    else:
        k = np.sum(M, axis = 0)
        coo = np.dot(M.T,M)
        B = np.nan_to_num(coo/(np.subtract.outer(k, -k)-coo))
    return B

# [Adamic_Adar 유사도]
def adamic_adar(M, projection):
    if projection == 0:
        k = np.sum(M, axis = 0)
        k = np.nan_to_num(1/np.log10(k))
        B = np.nan_to_num((M*k).dot(M.T))
    else:
        k = np.sum(M, axis = 1)
        k = np.nan_to_num(1/np.log10(k))
        B = np.nan_to_num((M.T*k).dot(M))
    return B

# [선호적 연결성 유사도]
def preferential_attachment(M, projection):
    if projection == 0:
        k = np.sum(M, axis = 1)
        B = np.nan_to_num(np.multiply.outer(k, k))
    else:
        k = np.sum(M, axis = 0)
        B = np.nan_to_num(np.multiply.outer(k, k))
    return B

# [리소스 할당 유사도]
def resource_allocation(M, projection):
    if projection == 0:
        k = np.sum(M, axis = 0)
        B = np.nan_to_num(np.nan_to_num(M/k).dot(M.T))
    else:
        k = np.sum(M, axis = 1)
        B = np.nan_to_num(np.nan_to_num(M.T/k).dot(M))
    return B


# [코사인 유사도]
def cosine_similarity(M, projection):
    if projection == 0:
        k = np.sum(M, axis = 1)
        coo = np.dot(M,M.T)
        B = np.nan_to_num(coo/(np.multiply.outer(k,k)**0.5))
    else:
        k = np.sum(M, axis = 0)
        coo = np.dot(M.T,M)
        B = np.nan_to_num(coo/(np.multiply.outer(k,k)**0.5))
    return B

# [Sorensen-Dice 유사도]
def sorensen(M, projection):
    if projection == 0:
        k = np.sum(M, axis = 1)
        coo = np.dot(M,M.T)
        B = np.nan_to_num(2*coo/(np.subtract.outer(k, -k)))
    else:
        k = np.sum(M, axis = 0)
        coo = np.dot(M.T,M)
        B = np.nan_to_num(2*coo/(np.subtract.outer(k, -k)))
    return B


# [Hub_Depressed_Index 유사도]
def hub_depressed_index(M, projection):
    if projection == 0:
        k = np.sum(M, axis = 1)
        coo = np.dot(M,M.T)
        B = np.nan_to_num(coo/np.maximum.outer(k.T,k))
    else:
        k = np.sum(M, axis = 0)
        coo = np.dot(M.T,M)
        B = np.nan_to_num(coo/np.maximum.outer(k.T,k))
    return B

# [Hub-Promoted Index 유사도]
def hub_promoted_index(M, projection):
    if projection == 0:
        k = np.sum(M, axis = 1)
        coo = np.dot(M,M.T)
        B = np.nan_to_num(coo/np.minimum.outer(k.T,k))
    else:
        k = np.sum(M, axis = 0)
        coo = np.dot(M.T,M)
        B = np.nan_to_num(coo/np.minimum.outer(k.T,k))
    return B

# [Taxonomy Network 유사도]
def taxonomy_network(M, projection):
    if projection == 0:
        k1 = np.sum(M, axis = 1)
        k2 = np.sum(M, axis = 0)
        B = np.nan_to_num(np.nan_to_num(M/k2).dot(M.T)/np.maximum.outer(k1.T,k1))
    else:
        k2 = np.sum(M, axis = 1)
        k1 = np.sum(M, axis = 0)
        B = np.nan_to_num(np.nan_to_num(M.T/k2).dot(M)/np.maximum.outer(k1.T,k1))
    return B

# [확률적 전파 유사도]
def probabilistic_spreading(M, projection):
    if projection == 0:
        k1 = np.sum(M, axis = 1)
        k2 = np.sum(M, axis = 0)
        B = np.nan_to_num(np.nan_to_num(M/k2).dot(M.T)/k1)
    else:
        k2 = np.sum(M, axis = 1)
        k1 = np.sum(M, axis = 0)
        B = np.nan_to_num(np.nan_to_num(M.T/k2).dot(M)/k1)
    return B

# [Pearson 상관 계수를 사용한 유사도]
def pearson(M, projection):
    if projection == 0:
        k = np.sum(M, axis = 1)
        coo = np.dot(M,M.T)
        N = M.shape[1]
        B = np.nan_to_num((coo-np.multiply.outer(k, k)/N)/np.nan_to_num(np.multiply.outer(np.sum((M.T-k/N).T**2, axis = 1)**0.5,np.sum((M.T-k/N).T**2, axis = 1)**0.5)))
    else:
        k = np.sum(M, axis = 0)
        coo = np.dot(M.T,M)
        N = M.shape[0]
        B = np.nan_to_num((coo-np.multiply.outer(k, k)/N)/np.nan_to_num(np.multiply.outer(np.sum((M-k/N)**2, axis = 0)**0.5,np.sum((M-k/N)**2, axis = 0)**0.5)))
    return B

# [Sapling 유사도]
def sapling(M, projection):
    if projection == 0:
        N = M.shape[1]
        k = np.sum(M, axis = 1)
        CO = np.dot(M,M.T)
        B = np.nan_to_num((1-(CO*(1-CO/k)+(k-CO.T).T*(1-(k-CO.T).T/(N-k))).T/(k*(1-k/N))).T*np.sign(((CO*N/k).T/k).T-1))
    else:
        N = M.shape[0]
        k = np.sum(M, axis = 0)
        CO = np.dot(M.T,M)
        B = np.nan_to_num((1-(CO*(1-CO/k)+(k-CO.T).T*(1-(k-CO.T).T/(N-k))).T/(k*(1-k/N))).T*np.sign(((CO*N/k).T/k).T-1))
    return B

# [Sapling 유사도를 Resource Allocation 방법을 사용하여 계산하는 함수]
def sapling_RA(M, projection):
    if projection == 0:
        u = np.sum(M,axis = 0)
        CO = np.dot(np.nan_to_num(M/u),M.T)
        N = np.sum(np.nan_to_num(1/u))
        k = np.sum(np.nan_to_num(M/u),axis = 1)
        B=np.nan_to_num((1-(CO*(1-CO/k)+(k-CO).T*(1-(k-CO).T/(N-k))).T/(k*(1-k/N))).T*np.sign(((CO*N/k).T/k).T-1))
    else:
        d = np.sum(M,axis = 1)
        CO = np.dot(np.nan_to_num(M.T/d),M)
        N = np.sum(np.nan_to_num(1/d))
        k = np.sum(np.nan_to_num(M.T/d),axis = 1)
        B=np.nan_to_num((1-(CO*(1-CO/k)+(k-CO).T*(1-(k-CO).T/(N-k))).T/(k*(1-k/N))).T*np.sign(((CO*N/k).T/k).T-1))
    return B



# [유사도 계산 함수]
# 'model' 인수에 따라 다양한 유사도 모델 선택
def similarity(M, model, projection):
    if model == "common_neighbors":
        B = common_neighbors(M, projection)
    elif model == "jaccard":
        B = jaccard(M, projection)
    elif model == "adamic_adar":
        B = adamic_adar(M, projection)
    elif model == "resource_allocation":
        B = resource_allocation(M, projection)
    elif model == "cosine_similarity":
        B = cosine_similarity(M, projection)
    elif model == "sorensen":
        B = sorensen(M, projection)
    elif model == "hub_depressed_index":
        B = hub_depressed_index(M, projection)
    elif model == "hub_promoted_index":
        B = hub_promoted_index(M, projection)
    elif model == "taxonomy_network":
        B = taxonomy_network(M, projection)
    elif model == "probabilistic_spreading":
        B = probabilistic_spreading(M, projection)
    elif model == "pearson":
        B = pearson(M, projection)
    else:
        B = sapling(M, projection)
    return B # 유사도 'B' 반환
        
