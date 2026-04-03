from Data.DataLoader import data_loader
import numpy as np


class MultiModalDataLoad:
    def __init__(self, data_path, dataset, sel_model_1, sel_model_2):
        self.DP = data_path
        self.DS = dataset
        self.SM1 = sel_model_1
        self.SM2 = sel_model_2

    def multiModelDataloader(self):
        Data = []
        Labels = []
        dl1 = data_loader(self.DP + self.DS + "/" + self.SM1 + "/", kind="Non")
        train_features, train_labels, val_features, val_labels, test_features, test_labels = dl1.feature_labels_loader()
        dl2 = data_loader(self.DP + self.DS + "/" + self.SM2 + "/", kind="Non")
        train_features1, train_labels1, val_features1, val_labels1, test_features1, test_labels1 = dl2.feature_labels_loader()
        Data.append([train_features, val_features, test_features])
        Data.append([train_features1, val_features1, test_features1])
        Labels.append(train_labels)
        Labels.append(val_labels)
        Labels.append(test_labels)
        Classes = sorted(np.unique(train_labels), key=int)
        Dim = train_features.shape[1] + train_features1.shape[1]
        return Data, Labels, Classes, Dim
