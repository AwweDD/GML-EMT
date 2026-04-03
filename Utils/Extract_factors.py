from sklearn import preprocessing
from os.path import join
import numpy as np
import pickle
import torch


class ExtractFactors:
    def __init__(self, K, Dim, Device, Labels, Classes, labeled_data, unlabeled_data):
        self.K = K
        self.Dim = Dim
        self.Device = Device
        self.Labels = Labels
        self.Classes = Classes
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data

    # knn classifier
    def knn(self, test_data, train_data, phase):
        label_train_labeled = self.Labels[0][self.labeled_data]
        K = self.K
        dist = torch.cdist(torch.tensor(test_data), torch.tensor(train_data))
        sort_dist, B = torch.topk(dist, k=K + 1, largest=False, dim=1)
        if phase == "train":
            sort_dist = sort_dist[:, 1:]
            B = B[:, 1:]
            sort_dist = sort_dist.cpu().numpy()
            B = B.cpu().numpy()
            sort_dist = 1 - preprocessing.normalize(sort_dist, axis=1)
            dist_ = {}
            for i in range(test_data.shape[0]):
                for k in range(K):
                    dist_[(i, B[i][k])] = [0, sort_dist[i][k]]
        else:
            sort_dist = sort_dist[:, :K]
            B = B[:, :K]
            sort_dist = sort_dist.cpu().numpy()
            B = B.cpu().numpy()
            sort_dist = 1 - preprocessing.normalize(sort_dist, axis=1)
            dist_ = {}
            for i in range(test_data.shape[0]):
                for k in range(K):
                    dist_[(i + len(label_train_labeled), B[i][k])] = [0, sort_dist[i][k]]
        return dist_

    def extractFeatureFactors(self, two_best_dec, dataset, ratio_lab, para):
        real_best_dec = two_best_dec[0]
        binary_best_dec = two_best_dec[1]
        #
        label_train_labeled = self.Labels[0][self.labeled_data]
        label_test_labeled = self.Labels[2]
        #
        """
            calculate CCD factor
        """
        fusion_train_fea_vec = real_best_dec[0][:]
        fusion_train_fea_labeled = fusion_train_fea_vec[self.labeled_data][:]
        #
        fusion_test_fea_vec = real_best_dec[2][:]
        #
        centers = []
        for i in range(len(self.Classes)):
            centers.append(np.median(fusion_train_fea_labeled[np.where(label_train_labeled == self.Classes[i])[0]], axis=0).tolist())
        #
        dist_train = torch.cdist(torch.tensor(fusion_train_fea_labeled), torch.tensor(np.array(centers)), p=3).cpu().numpy()
        dist_train = np.exp(dist_train) / np.sum(np.exp(dist_train), axis=1)[:, np.newaxis]
        dist_train = 1 - preprocessing.normalize(dist_train, axis=1)
        #
        dist_test = torch.cdist(torch.tensor(fusion_test_fea_vec), torch.tensor(np.array(centers)), p=3).cpu().numpy()
        dist_test = np.exp(dist_test) / np.sum(np.exp(dist_test), axis=1)[:, np.newaxis]
        dist_test = 1 - preprocessing.normalize(dist_test, axis=1)
        #
        # define template
        create_v_file = []
        dict_tem = {
            'var_id': 0,
            'is_easy': None,
            'is_evidence': None,
            'label': None,
            'true_label': None,
            'feature_set': {}
        }
        feature_set = {int(category): None for category in self.Classes}
        #
        for i in range(dist_train.shape[0]):
            new_dict_tem = dict_tem.copy()
            new_fea_set = feature_set.copy()
            new_dict_tem['var_id'] = i
            new_dict_tem['is_easy'] = True
            new_dict_tem['is_evidence'] = True
            new_dict_tem['label'] = int(label_train_labeled[i])
            new_dict_tem['true_label'] = int(label_train_labeled[i])
            for j in range(len(new_fea_set)):
                new_fea_set[j] = [0, dist_train[i][j]]
            new_fea_set[j + 1] = [0.5, 0]
            new_dict_tem['feature_set'] = new_fea_set
            create_v_file.append(new_dict_tem)
        for i in range(dist_test.shape[0]):
            new_dict_tem = dict_tem.copy()
            new_fea_set = feature_set.copy()
            new_dict_tem['var_id'] = i + dist_train.shape[0]
            new_dict_tem['is_easy'] = False
            new_dict_tem['is_evidence'] = False
            new_dict_tem['label'] = -1
            new_dict_tem['true_label'] = int(label_test_labeled[i])
            for j in range(len(new_fea_set)):
                new_fea_set[j] = [0, dist_test[i][j]]
            new_fea_set[j + 1] = [0.5, 0]
            new_dict_tem['feature_set'] = new_fea_set
            create_v_file.append(new_dict_tem)
        pickle.dump(create_v_file, open(join("./datapath", 'v_{}_{}_{}.pkl'.format(dataset, ratio_lab, para)), 'wb+'))
        """
            calculate the KNN factor
        """
        fusion_train_fea_vec = binary_best_dec[0][:]
        fusion_train_fea_labeled = fusion_train_fea_vec[self.labeled_data][:]
        #
        fusion_test_fea_vec = binary_best_dec[2][:]
        create_f_file = []
        dict_tem_f = {
            'feature_id': 0,
            'feature_type': "",
            'feature_name': None,
            'parameterize': None,
            'monotonicity': None,
            'weight': {},
            'Association_category': None
        }
        for i in range(len(self.Classes)):
            feature_set_f_ccd = {}
            new_dict_tem_f = dict_tem_f.copy()
            new_dict_tem_f['feature_id'] = i
            new_dict_tem_f['feature_type'] = 'unary_feature'
            new_dict_tem_f['feature_name'] = 'center_distance_' + str(i)
            new_dict_tem_f['parameterize'] = 1
            new_dict_tem_f['monotonicity'] = True
            for j in range(dist_train.shape[0]):
                dist = dist_train[j][i]
                feature_set_f_ccd[j] = [0, dist]
            for j in range(dist_test.shape[0]):
                dist = dist_test[j][i]
                feature_set_f_ccd[j + dist_train.shape[0]] = [0, dist]
            new_dict_tem_f['weight'] = feature_set_f_ccd
            new_dict_tem_f['Association_category'] = i
            create_f_file.append(new_dict_tem_f)
        #
        knn_factor_train = self.knn(fusion_train_fea_labeled, fusion_train_fea_labeled, "train")
        #
        knn_factor_test = self.knn(fusion_test_fea_vec, fusion_train_fea_labeled, "test")
        knn_factor_train.update(knn_factor_test)
        #
        new_dict_tem_f = dict_tem_f.copy()
        new_dict_tem_f['feature_id'] = i + 1
        new_dict_tem_f['feature_type'] = 'binary_feature'
        new_dict_tem_f['feature_name'] = 'knn_pair'
        new_dict_tem_f['parameterize'] = 1
        new_dict_tem_f['monotonicity'] = True
        new_dict_tem_f['weight'] = knn_factor_train
        new_dict_tem_f['Association_category'] = -1
        create_f_file.append(new_dict_tem_f)
        pickle.dump(create_f_file, open(join("./datapath/", 'f_{}_{}_{}.pkl'.format(dataset, ratio_lab, para)), 'wb+'))
