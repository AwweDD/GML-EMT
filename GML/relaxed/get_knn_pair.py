import torch
import numpy as np


# Calculate the knn_pair to each class center of every data
def get_knn_pair(K, train_features, train_labels, val_features, val_labels, test_features, test_labels):
    distribution_all = np.concatenate((train_features, test_features), axis=0)
    labels_all = np.concatenate((train_labels, test_labels), axis=0)
    # Get pairwise_distances for all
    c_sim1 = torch.cdist(torch.tensor(np.array(train_features)), torch.tensor(np.array(train_features)))
    c_sim2 = torch.cdist(torch.tensor(np.array(test_features)), torch.tensor(np.array(train_features)))
    # Get knn for all
    sorted_dist, B = torch.sort(c_sim1)
    sorted_dist = sorted_dist.cpu().numpy().tolist()
    B = B.cpu().numpy().tolist()
    pair1 = [[[i, B[i][j + 1], sorted_dist[i][j + 1]] for j in range(K)] for i in range(len(train_features))]
    #
    sorted_dist, B = torch.sort(c_sim2)
    sorted_dist = sorted_dist.cpu().numpy().tolist()
    B = B.cpu().numpy().tolist()
    pair2 = [[[i + len(train_features), B[i][j], sorted_dist[i][j]] for j in range(K)] for i in range(len(test_features))]
    #
    pair = np.concatenate((pair1, pair2), axis=0)
    pair_len = pair.shape[0]
    pair = pair.tolist()
    # Combine all pair
    for i in range(pair_len):
        pair[i].insert(0, i)
        pair[i].insert(1, labels_all[i])
    # Create the header of csv
    header = ["data", "label"]
    for i in range(K):
        header.append("nearest_neighbor_{}".format(i))
    pair.insert(0, header)
    return pair
