import pandas as pd
import numpy as np
import re


class data_loader:
    def __init__(self, Data_dir, kind="Non"):
        self.Data_dir = Data_dir
        self.Kind = kind

    def feature_labels_loader(self):
        # Load train and test features and labels
        Data_dir = self.Data_dir
        pattern = r'\d+'
        # =============================Train_Data============================
        train_fea = pd.read_csv(Data_dir + 'train_features.csv', header=None)
        train_features = np.array(train_fea.values)
        # =======
        train_lab = pd.read_csv(Data_dir + 'train_targets.csv', header=None)
        train_labels = []
        for item in train_lab.values:
            lab = re.findall(pattern, str(item))[0]
            train_labels.append(lab)
        train_labels = np.array(train_labels)
        # =============================Val_Data==============================
        val_fea = pd.read_csv(Data_dir + 'val_features.csv', header=None)
        val_features = np.array(val_fea.values)
        # =======
        val_lab = pd.read_csv(Data_dir + 'val_targets.csv', header=None)
        val_labels = []
        for item in val_lab.values:
            lab = re.findall(pattern, str(item))[0]
            val_labels.append(lab)
        val_labels = np.array(val_labels)
        # # ============================Test_Data============================
        test_fea = pd.read_csv(Data_dir + 'test_features.csv', header=None)
        test_features = np.array(test_fea.values)
        # =======
        test_labs = pd.read_csv(Data_dir + 'test_targets.csv', header=None)
        test_labels = []
        for item in test_labs.values:
            lab = re.findall(pattern, str(item))[0]
            test_labels.append(lab)
        test_labels = np.array(test_labels)
        # ========================data normalization=========================
        if self.Kind == "Linear_Norm":
            """
            Linear Normalization
            """
            # train
            min_ = np.min(train_features, axis=0)
            max_ = np.max(train_features, axis=0)
            train_features = (train_features - min_) / (max_ - min_)
            # val
            min_ = np.min(val_features, axis=0)
            max_ = np.max(val_features, axis=0)
            val_features = (val_features - min_) / (max_ - min_)
            # test
            min_ = np.min(test_features, axis=0)
            max_ = np.max(test_features, axis=0)
            test_features = (test_features - min_) / (max_ - min_)
        elif self.Kind == "Z_Norm":
            """
            Z-score Normalization
            """
            # train
            mean_ = np.mean(train_features, axis=0)
            std_ = np.std(train_features, axis=0)
            train_features = (train_features - mean_) / std_
            # val
            mean_ = np.mean(val_features, axis=0)
            std_ = np.std(val_features, axis=0)
            val_features = (val_features - mean_) / std_
            # test
            mean_ = np.mean(test_features, axis=0)
            std_ = np.std(test_features, axis=0)
            test_features = (test_features - mean_) / std_
        else:
            pass

        return train_features, train_labels, val_features, val_labels, test_features, test_labels
