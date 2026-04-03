from GmlMultiLabel.gml import GML
import pickle

if __name__ == '__main__':
    ratio = 1
    class_num = 9
    Datasets = ["NCT-CRC-HE-100K"]
    Models = ['densenet121', 'wide_resnet50_2']
    #
    for i in range(1):
        with open("./Data/f_" + Datasets[0] + "_" + str(ratio) + "_" + str(i) + ".pkl", 'rb') as v:
            features = pickle.load(v)
        with open("./Data/v_" + Datasets[0] + "_" + str(ratio) + "_" + str(i) + ".pkl", 'rb') as f:
            variables = pickle.load(f)
        graph = GML.initial("example.config", variables, features, class_num)
        # inference
        graph.inference()
