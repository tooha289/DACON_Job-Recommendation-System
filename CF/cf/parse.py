import argparse

# [argparse]
# 주로 스크립트 또는 프로그램을 실행할 때 '사용자로부터 입력'을 받거나 스크립트에 '설정값을 전달'하기 위해 사용
# argparse를 사용하면 사용자와 스크립트 간의 상호작용을 단순하게 만들고 사용자에게 명령행에서 어떤 옵션 및 매개변수를 전달해야 하는지 알려준다.

def parse_args():
    parser = argparse.ArgumentParser(description="Go lightGCN")
    parser.add_argument('--gamma', type=float, default=0.2,
                        help="value of the gamma parameter used to combine user-based and item-based approaches")
    parser.add_argument('--dataset', type=str, default='dacon',
                        help="available datasets: [dacon, export, amazon-product, gowalla, yelp2018, amazon-book]")
    parser.add_argument('--similarity', type=str, default="pearson",
                        help="available similarities: [common_neighbors, jaccard, adamic_adar, resource_allocation, cosine_similarity, sorensen, hub_depressed_index, hub_promoted_index, taxonomy_network, probabilistic_spreading, pearson, sapling_similarity]")
    parser.add_argument('--test', type=str, default="test",
                        help="data to apply the model to: [validation, test]")
    return parser.parse_args()

# 이 함수를 호출하면 사용자가 지정한 인수들을 포함한 객체가 반환됩니다.