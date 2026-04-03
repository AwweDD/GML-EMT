from sklearn.metrics import accuracy_score, roc_auc_score, f1_score, confusion_matrix, precision_score
import torch.nn as nn
import numpy as np
import torch


class myUtils:
    def __init__(self, Data, Labels, Classes, Device, Dim, N, K, alpa, beta, labeled_data, unlabeled_data):
        self.Data = Data
        self.Labels = Labels
        self.Classes = Classes
        self.Device = Device
        self.Dim = Dim
        self.N = N
        self.K = K
        self.alpa = alpa
        self.beta = beta
        self.labeled_data = labeled_data
        self.unlabeled_data = unlabeled_data

    # knn classifier
    def knn(self, lab_data, train_data, train_label, K):
        dist = torch.cdist(torch.tensor(lab_data), torch.tensor(train_data))
        sort_dist, B = torch.topk(dist, k=K, largest=False, dim=1)
        w = 1 / sort_dist.cpu().numpy()
        y_pred_label = []
        y_pred_label_pro = []
        for i in range(lab_data.shape[0]):
            pro_ = []
            labels = train_label[B[i]]
            for j in range(len(self.Classes)):
                pro_.append(np.sum(w[i][np.where((labels == self.Classes[j]))[0]]) / np.sum(w[i]))
            y_pred_label.append(str(np.argmax(pro_)))
            y_pred_label_pro.append(pro_)
        return np.array(y_pred_label), np.array(y_pred_label_pro)

    def cal_kl_(self, p, q):
        _ = self.N
        p = np.clip(p, 1e-10, 1)
        q = np.clip(q, 1e-10, 1)
        kl = np.mean(np.sum(p * np.log(p / q)))
        return kl

    def cal_model_consistency(self, acc, train_unlabeled_pro):
        all_kl = []
        for i in range(len(acc)):
            kl = []
            for j in range(len(acc)):
                kl_i = self.cal_kl_(train_unlabeled_pro[i], train_unlabeled_pro[j])
                kl.append(kl_i)
            all_kl.append(np.sum((acc - np.mean(acc)) * (kl - np.mean(kl))))
        all_kl = (np.array(all_kl) - np.min(all_kl) + 1e-10) / (np.max(all_kl) - np.min(all_kl) + 1e-10)
        return all_kl

    def cal_data_consistency(self, pro_cls, fusion_para, Coding):
        """
        add_noise_data
        """
        label_train_labeled = self.Labels[0][self.labeled_data]
        m = nn.Softmax(dim=1)
        #
        if Coding == 'real':
            all_feature_vectors = []
            for j in range(len(self.Data[0])):
                con_data = np.concatenate((self.Data[0][j], self.Data[1][j]), axis=1)
                gaussian_noise = np.random.uniform(0, 0.2, con_data.shape)
                all_feature_vectors.append((con_data + gaussian_noise) * fusion_para)
            #
            fusion_train_fea_vec = all_feature_vectors[0][:]
            fusion_train_fea_labeled = fusion_train_fea_vec[self.labeled_data][:]
            fusion_train_fea_unlabeled = fusion_train_fea_vec[self.unlabeled_data][:]
            centers = []
            for i in range(len(self.Classes)):
                centers.append(np.median(fusion_train_fea_labeled[np.where(label_train_labeled == self.Classes[i])[0]], axis=0).tolist())
            dist_ = torch.cdist(torch.tensor(fusion_train_fea_unlabeled), torch.tensor(np.array(centers))).cpu().numpy()
            dist_ = np.exp(dist_) / np.sum(np.exp(dist_), axis=1)[:, np.newaxis]
            pro_cls_ = np.array(m(torch.from_numpy(-dist_)).cpu().numpy().tolist())
        else:
            all_feature_vectors = []
            for j in range(len(self.Data[0])):
                con_data = np.concatenate((self.Data[0][j], self.Data[1][j]), axis=1)
                gaussian_noise = np.random.uniform(0, 0.2, con_data.shape)
                all_feature_vectors.append((con_data + gaussian_noise) * fusion_para)
            # calculate the accuracy of validation samples
            fusion_train_fea_vec = all_feature_vectors[0][:]
            fusion_train_fea_labeled = fusion_train_fea_vec[self.labeled_data][:]
            fusion_train_fea_unlabeled = fusion_train_fea_vec[self.unlabeled_data][:]
            # unlabeled samples prediction ratio
            pred_labs_, pred_pro_ = self.knn(fusion_train_fea_unlabeled, fusion_train_fea_labeled, label_train_labeled, self.K)
            pro_cls_ = np.array(m(torch.from_numpy(pred_pro_)).cpu().numpy().tolist())
        #
        kl = self.cal_kl_(pro_cls, pro_cls_)
        return kl

    # calculate the objs for all individuals
    def cal_two_task_obj(self, fusion_feature_vectors, Coding):
        label_train_labeled = self.Labels[0][self.labeled_data]
        m = nn.Softmax(dim=1)
        right_index = np.zeros((1, len(self.Labels[1])))
        if Coding == 'real':
            """
                calculate the objective function values for the first task
            """
            fusion_train_fea_vec = fusion_feature_vectors[0][:]
            fusion_train_fea_labeled = fusion_train_fea_vec[self.labeled_data][:]
            fusion_train_fea_unlabeled = fusion_train_fea_vec[self.unlabeled_data][:]
            #
            centers = []
            for i in range(len(self.Classes)):
                # find the median value of each element of the labeled train features in fusion feature vectors
                centers.append(np.median(fusion_train_fea_labeled[np.where(label_train_labeled == self.Classes[i])[0]], axis=0).tolist())
            #
            dist = torch.cdist(torch.tensor(fusion_feature_vectors[1][:]), torch.tensor(np.array(centers)), p=3).cpu().numpy()
            pred_labs = np.argmin(dist, axis=1).astype(str)
            #
            acc = accuracy_score(self.Labels[1], pred_labs)
            #
            right_index[:, np.where(self.Labels[1] == pred_labs)[0]] = 1
            # unlabeled samples prediction ratio
            dist_ = torch.cdist(torch.tensor(fusion_train_fea_unlabeled), torch.tensor(np.array(centers))).cpu().numpy()
            dist_ = np.exp(dist_) / np.sum(np.exp(dist_), axis=1)[:, np.newaxis]
            #
            pro_cls = m(torch.from_numpy(-dist_)).cpu().numpy().tolist()
            result = [acc, np.array(pro_cls), right_index]
        else:
            """
                calculate the objective function values for the second task
            """
            # calculate the accuracy of validation samples
            fusion_train_fea_vec = fusion_feature_vectors[0][:]
            fusion_train_fea_labeled = fusion_train_fea_vec[self.labeled_data][:]
            fusion_train_fea_unlabeled = fusion_train_fea_vec[self.unlabeled_data][:]
            #
            pred_labs, pred_pro = self.knn(fusion_feature_vectors[1][:], fusion_train_fea_labeled, label_train_labeled, self.K)
            #
            acc = accuracy_score(self.Labels[1], pred_labs)
            #
            right_index[:, np.where(self.Labels[1] == pred_labs)[0]] = 1
            # unlabeled samples prediction ratio
            pred_labs_, pred_pro_ = self.knn(fusion_train_fea_unlabeled, fusion_train_fea_labeled, label_train_labeled, self.K)
            pro_cls = m(torch.from_numpy(pred_pro_)).cpu().numpy().tolist()
            result = [acc, np.array(pro_cls), right_index]
        return result

    def cal_objects_train(self, PopDecs, Coding):
        acc = []
        right_index = []
        data_consistency = []
        train_unlabeled_pro = []
        for i in range(len(PopDecs)):
            fusion_para = PopDecs[i][:]
            fusion_feature_vectors = []
            for j in range(len(self.Data[0])):
                fusion_feature_vectors.append(np.concatenate((self.Data[0][j], self.Data[1][j]), axis=1) * fusion_para)
            #
            results = self.cal_two_task_obj(fusion_feature_vectors, Coding)
            acc.append(results[0])
            train_unlabeled_pro.append(results[1])
            right_index.append(results[2])
            #
            kl = self.cal_data_consistency(results[1], fusion_para, Coding)
            data_consistency.append(kl)
        #
        acc_s = np.array(acc)
        right_index = np.array(right_index)
        data_consistency = (np.array(data_consistency) - np.min(data_consistency)) / (np.max(data_consistency) - np.min(data_consistency))
        model_consistency = self.cal_model_consistency(acc_s, np.array(train_unlabeled_pro))
        #
        Pop_task_Objs = np.vstack((1 - acc_s, self.alpa * data_consistency + self.beta * model_consistency)).T
        Pop_task_Fits = acc_s - self.alpa * data_consistency - self.beta * model_consistency
        return Pop_task_Objs, Pop_task_Fits, right_index

    # calculate the objs for all individuals
    def cal_two_task_obj_test(self, fusion_feature_vectors, Coding):
        label_train_labeled = self.Labels[0][self.labeled_data]
        label_test = self.Labels[2]
        m = nn.Softmax(dim=1)
        if Coding == 'real':
            """
                calculate the objective function values for the first task
            """
            fusion_train_fea_vec = fusion_feature_vectors[0][:]
            fusion_train_fea_labeled = fusion_train_fea_vec[self.labeled_data][:]
            #
            fusion_test_fea_vec = fusion_feature_vectors[2][:]
            #
            centers = []
            for i in range(len(self.Classes)):
                # find the median value of each element of the labeled train features in fusion feature vectors
                centers.append(np.median(fusion_train_fea_labeled[np.where(label_train_labeled == self.Classes[i])[0]], axis=0).tolist())
            #
            dist = torch.cdist(torch.tensor(fusion_test_fea_vec), torch.tensor(np.array(centers)), p=3).cpu().numpy()
            dist = dist / np.sum(dist, axis=1)[:, np.newaxis]
            pred_labs = np.argmin(dist, axis=1).astype(str)
            # test acc
            acc = accuracy_score(label_test, pred_labs)
            f1 = f1_score(label_test, pred_labs, average='macro')
            precision = precision_score(label_test, pred_labs, average='macro')
            #
            cm = confusion_matrix(label_test, pred_labs)
            n_classes = cm.shape[0]
            sensitivity_per_class = []
            for i in range(n_classes):
                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - tp
                sensitivity_i = tp / (tp + fn) if (tp + fn) != 0 else 0.0
                sensitivity_per_class.append(sensitivity_i)
            sensitivity = np.mean(sensitivity_per_class)
            #
            specificity_per_class = []
            for i in range(n_classes):
                tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
                fp = np.sum(cm[:, i]) - cm[i, i]
                specificity_i = tn / (tn + fp) if (tn + fp) != 0 else 0.0
                specificity_per_class.append(specificity_i)
            #
            specificity = np.mean(specificity_per_class)
            #
            mean_class_recall = np.mean(np.diag(cm) / np.sum(cm, axis=1))
            #
            pro_cls = m(torch.from_numpy(-dist)).cpu().numpy().tolist()
            result = [acc, f1, mean_class_recall, precision, sensitivity, specificity, np.array(pro_cls)]
        else:
            """
                calculate the objective function values for the second task
            """
            # calculate the accuracy of validation samples
            fusion_train_fea_vec = fusion_feature_vectors[0][:]
            fusion_train_fea_labeled = fusion_train_fea_vec[self.labeled_data][:]
            #
            fusion_test_fea_vec = fusion_feature_vectors[2][:]
            #
            pred_labs, pred_pro = self.knn(fusion_test_fea_vec, fusion_train_fea_labeled, label_train_labeled, self.K)
            # test acc
            acc = accuracy_score(label_test, pred_labs)
            f1 = f1_score(label_test, pred_labs, average='macro')
            precision = precision_score(label_test, pred_labs, average='macro')
            #
            cm = confusion_matrix(label_test, pred_labs)
            n_classes = cm.shape[0]
            sensitivity_per_class = []
            for i in range(n_classes):
                tp = cm[i, i]
                fn = np.sum(cm[i, :]) - tp
                sensitivity_i = tp / (tp + fn) if (tp + fn) != 0 else 0.0
                sensitivity_per_class.append(sensitivity_i)
            sensitivity = np.mean(sensitivity_per_class)
            #
            specificity_per_class = []
            for i in range(n_classes):
                tn = np.sum(cm) - np.sum(cm[i, :]) - np.sum(cm[:, i]) + cm[i, i]
                fp = np.sum(cm[:, i]) - cm[i, i]
                specificity_i = tn / (tn + fp) if (tn + fp) != 0 else 0.0
                specificity_per_class.append(specificity_i)
            #
            specificity = np.mean(specificity_per_class)
            #
            mean_class_recall = np.mean(np.diag(cm) / np.sum(cm, axis=1))
            #
            pro_cls = m(torch.from_numpy(pred_pro)).cpu().numpy().tolist()
            result = [acc, f1, mean_class_recall, precision, sensitivity, specificity, np.array(pro_cls)]
        return result

    # calculate the objs for all individuals
    def cal_objects_test(self, fusion_feature_vectors, Code):
        result = self.cal_two_task_obj_test(fusion_feature_vectors, Code)
        acc = result[0]
        f1 = result[1]
        MCR = result[2]
        precision = result[3]
        sensitivity = result[4]
        specificity = result[5]
        if len(self.Classes) == 2:
            A = result[6]
            B = A[:, 1]
            auc = roc_auc_score(self.Labels[2].tolist(), B)
        else:
            auc = roc_auc_score(self.Labels[2].tolist(), result[6], multi_class='ovr', average='macro')
        return acc, f1, MCR, precision, sensitivity, specificity, auc
