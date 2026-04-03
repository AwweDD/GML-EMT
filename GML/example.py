from GmlMultiLabel.gml import GML
import pickle

if __name__ == '__main__':
    ratio = 1
    class_num = 9
    Datasets = ["NCT-CRC-HE-100K"]
    Models = ['densenet121', 'wide_resnet50_2']
    #
    with open("/home/15t/lww/Data_paper2/pkl_test/f_" + Datasets[0] + "_" + str(ratio) + "_" + str(20000) + ".pkl", 'rb') as v:
        features = pickle.load(v)
    with open("/home/15t/lww/Data_paper2/pkl_test/v_" + Datasets[0] + "_" + str(ratio) + "_" + str(20000) + ".pkl", 'rb') as f:
        variables = pickle.load(f)
    graph = GML.initial("example.config", variables, features, class_num)
    # inference
    graph.inference()
