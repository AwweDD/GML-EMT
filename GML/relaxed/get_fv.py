import numpy as np
from sklearn import preprocessing


def get_f_pkl(num_class, distance_center_all, knn_pair, feature_names, parameterize):
    # Add factors
    models_1_pro = []
    factors = []
    f_idx = 0  # index of factor
    for feature_name in feature_names:
        if feature_name == "model_1_pro":
            if num_class == 2:
                for i in range(len(models_1_pro)):
                    feature = {}
                    feature['feature_id'] = f_idx
                    feature['feature_type'] = 'unary_feature'
                    feature['feature_name'] = '{}_{}'.format(feature_name, i)
                    feature['parameterize'] = parameterize
                    feature['monotonicity'] = True
                    feature['weight'] = {}
                    for j in range(len(models_1_pro[i])):
                        feature['weight'][j] = [0, float(models_1_pro[i][j])]
                    feature['Association_category'] = 1
                    factors.append(feature)
                    f_idx += 1
            else:
                for c in range(num_class):  # one feature for each category
                    feature = {}
                    feature['feature_id'] = f_idx
                    feature['feature_type'] = 'unary_feature'
                    feature['feature_name'] = '{}_{}'.format(feature_name, c)
                    feature['parameterize'] = parameterize
                    feature['monotonicity'] = True
                    feature['weight'] = {}
                    for i in range(len(distance_center_all)):
                        feature['weight'][i] = [0, float(distance_center_all[i][c])]
                    feature['Association_category'] = c
                    factors.append(feature)
                    f_idx += 1
        elif feature_name == "center_distance":  # 类中心距离
            distance_center_all = 1 - preprocessing.normalize(distance_center_all, axis=1)
            for c in range(num_class):  # one feature for each category
                feature = {}
                feature['feature_id'] = f_idx
                feature['feature_type'] = 'unary_feature'
                feature['feature_name'] = '{}_{}'.format(feature_name, c)
                feature['parameterize'] = parameterize
                feature['monotonicity'] = True
                feature['weight'] = {}
                for i in range(len(distance_center_all)):
                    feature['weight'][i] = [0, float(distance_center_all[i][c])]
                feature['Association_category'] = c
                factors.append(feature)
                f_idx += 1
        elif feature_name == "knn_pair":  # knn对
            feature = {}
            feature['feature_id'] = f_idx
            feature['feature_type'] = 'binary_feature'
            feature['feature_name'] = feature_name
            feature['parameterize'] = parameterize  # parameterize
            feature['monotonicity'] = True
            feature['weight'] = {}
            feature['Association_category'] = -1
            #
            a_s = []
            for i in range(len(knn_pair)):
                a_ = []
                for idx in range(len(knn_pair[i])):
                    a_.append(knn_pair[i][idx][-1])
                    a_s.append(a_)
            a_s = 1 - preprocessing.normalize(a_s, axis=1)
            for i in range(len(knn_pair)):
                for idx in range(len(knn_pair[i])):
                    feature['weight'][(int(knn_pair[i][idx][0]), int(knn_pair[i][idx][1]))] = [0, float(a_s[i][idx])]
            factors.append(feature)
            f_idx += 1
    return factors


def get_v_pkl(num_class, distance_center_all, knn_pair, labels_all, feature_names, parameterize):
    models_1_pro = []
    targets = np.concatenate((labels_all[0], labels_all[2]), axis=0)
    # Create variables list
    variables = []
    for i in range(len(targets)):  # one variable for each data
        flag = True if i < len(labels_all[0]) else False
        variable = {}
        variable['var_id'] = i
        variable['is_easy'] = True if flag else False
        variable['is_evidence'] = True if flag else False
        variable['label'] = int(targets[i]) if flag else -1
        variable['true_label'] = int(targets[i])
        variable['feature_set'] = {}
        variables.append(variable)
    # Add factors
    for n_idx in range(len(feature_names)):
        if feature_names[n_idx] == "model_1_pro":
            features = np.array(models_1_pro)
        elif feature_names[n_idx] == "center_distance":
            features = np.array(1 - preprocessing.normalize(distance_center_all, axis=1))
        elif feature_names[n_idx] == "knn_pair":
            features = np.array(knn_pair)
        else:
            continue
        for i in range(len(variables)):
            if feature_names[n_idx] == "model_1_pro":
                for j in range(len(features)):
                    variables[i]['feature_set'][len(variables[i]['feature_set'])] = [0, float(features[j][i])]
            elif feature_names[n_idx] == "center_distance":
                for j in range(len(features[i])):
                    variables[i]['feature_set'][len(variables[i]['feature_set'])] = [0.5, float(features[i][j])]
            elif feature_names[n_idx] == "knn_pair":
                variables[i]['feature_set'][len(variables[i]['feature_set'])] = [0.5, 0]
    return variables


def get_fv(distance_center_all, knn_pair, labels_all, num_classes):
    distance_center_all = distance_center_all[1:]
    knn_pair = knn_pair[1:]
    for i in range(len(distance_center_all)):
        distance_center_all[i] = distance_center_all[i][2:]
        knn_pair[i] = knn_pair[i][2:]
    f_ = get_f_pkl(num_classes, distance_center_all, knn_pair, ['center_distance', 'knn_pair'], 1)
    v_ = get_v_pkl(num_classes, distance_center_all, knn_pair, labels_all, ['center_distance', 'knn_pair'], 1)
    # 当前数据距离各类中心的距离，当前数据的knn pair
    return f_, v_
