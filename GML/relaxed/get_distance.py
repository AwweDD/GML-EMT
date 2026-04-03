import numpy as np
import torch
from sklearn.neighbors import NearestCentroid


def get_distance(train_features, train_labels, val_features, val_labels, test_features, test_labels, num_classes, metric):
    features_ = np.concatenate((train_features, val_features), axis=0)
    labels_ = np.concatenate((train_labels, val_labels), axis=0)
    centers = np.array(NearestCentroid(metric=metric).fit(train_features, train_labels).centroids_)
    #
    train_distance_to_center = torch.cdist(torch.tensor(train_features), torch.tensor(centers)).cpu().numpy().tolist()
    # val_distance_to_center = torch.cdist(torch.tensor(val_features), torch.tensor(centers)).cpu().numpy().tolist()
    test_distance_to_center = torch.cdist(torch.tensor(test_features), torch.tensor(centers)).cpu().numpy().tolist()
    # Insert path and label of data to csv
    for i in range(len(train_distance_to_center)):
        train_distance_to_center[i].insert(0, i)
        train_distance_to_center[i].insert(1, train_labels[i])
    # for i in range(len(val_distance_to_center)):
    #     val_distance_to_center[i].insert(0, len(train_distance_to_center) + i)
    #     val_distance_to_center[i].insert(1, val_labels[i])
    for i in range(len(test_distance_to_center)):
        test_distance_to_center[i].insert(0, len(train_distance_to_center) + i)
        test_distance_to_center[i].insert(1, test_labels[i])
    #
    header = ["data", "label"]
    for i in range(num_classes):
        header.append("class_{:0>2d}_distance".format(i))
    #
    distance_center_all = np.concatenate((train_distance_to_center, test_distance_to_center)).tolist()
    distance_center_all.insert(0, header)
    return distance_center_all
