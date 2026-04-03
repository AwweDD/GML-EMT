from GmlMultiLabel.relaxed.get_distance import get_distance
from GmlMultiLabel.relaxed.get_knn_pair import get_knn_pair
from GmlMultiLabel.relaxed.get_fv import get_fv


def get_gml_need_(data, num_classes, K):
    center_distance = get_distance(data[0], data[1], data[2], data[3], data[4], data[5], num_classes, "euclidean")
    knn_pair = get_knn_pair(K, data[0], data[1], data[2], data[3], data[4], data[5])
    f_, v_ = get_fv(center_distance, knn_pair, [data[1], data[3], data[5]], num_classes)
    return f_, v_
