from Data.MultiModalData import MultiModalDataLoad
from EA_Fusion.PopulationInitialize import *
from EA_Fusion.TournamentSelection import *
from EA_Fusion.Evolutioin import *
from Utils import Extract_factors
from Utils import MyUtils
import numpy as np
import torch
import time

if __name__ == '__main__':
    start_time = time.time()
    """
        Prepare
    """
    Device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    Datasets = ["SkinCancerMNISTHAM10000"]
    Models = ['densenet121', 'wide_resnet50_2']
    #
    sel_labeled_ratios = [0.2]
    for dataset in Datasets:
        for sel_labeled_ratio in sel_labeled_ratios:
            DataPath = "data_path"
            #
            """
                Data Loading
            """
            MMD = MultiModalDataLoad(DataPath, dataset, Models[0], Models[1])
            Data, Labels, Classes, Dim = MMD.multiModelDataloader()
            """
                assign the labeled and unlabeled samples
            """
            labeled_samples = []
            unlabeled_samples = []
            for i in range(len(Classes)):
                i_cls = np.where(Labels[0] == Classes[i])[0].tolist()
                np.random.shuffle(i_cls)
                sel_sam = int(len(i_cls) * sel_labeled_ratio)
                labeled_samples = labeled_samples + i_cls[:sel_sam]
                unlabeled_samples = unlabeled_samples + i_cls[sel_sam:]
            labeled_samples = np.array(labeled_samples)
            unlabeled_samples = np.array(unlabeled_samples)
            """
                the evolution process
            """
            # the number of iterations
            Iter = 5
            # population size
            N = 20
            # the number of sub-problem
            M = 2
            # the value of K in knn
            K = 7
            # the number of task
            Task_num = 2
            #
            Beta = 0.025
            #
            Alpa = 0.025
            #
            CalObjs = MyUtils.myUtils(Data, Labels, Classes, Device, Dim, N, K, Alpa, Beta, labeled_samples, unlabeled_samples)
            # generation the initial population
            Populations = PopulationGeneration(Dim, Task_num).generation(CalObjs)
            """
                Training stage...
            """
            fea_wei = np.zeros(Dim)
            for t in range(Iter):
                for p in range(Task_num):
                    Population = Populations[p]
                    # parents selection
                    Parents = ESelection(N).TournamentSelection(Population.Fits)
                    # evolution
                    Offspring = PopEvolution(Dim).Evolve(Population.Coding, Population.Decs[Parents], CalObjs, fea_wei, t)
                    # update the feature weight
                    a = (Offspring.Decs - Population.Decs[Parents])
                    b = (Offspring.Fits - Population.Fits[Parents]).reshape(-1, 1)
                    c = a * b
                    d = np.mean(c, axis=0)
                    fea_wei = fea_wei + d
                    fea_wei = (fea_wei - np.min(fea_wei) + 1e-10) / (np.max(fea_wei) - np.min(fea_wei) + 1e-10)
                    # population update
                    AllDecs = np.concatenate((Population.Decs, Offspring.Decs), axis=0)
                    AllObjs = np.concatenate((Population.Objs, Offspring.Objs), axis=0)
                    AllFits = np.concatenate((Population.Fits, Offspring.Fits), axis=0)
                    sorted_indices = np.argsort(-AllFits)
                    Population.Decs = AllDecs[sorted_indices[:N]][:]
                    Population.Objs = AllObjs[sorted_indices[:N]][:]
                    Population.Fits = AllFits[sorted_indices[:N]][:]
                    #
                    print(t, 1 - np.min(Population.Objs, axis=0))
            """
                Test stage..............
            """
            ef = Extract_factors.ExtractFactors(K, Dim, Device, Labels, Classes, labeled_samples, unlabeled_samples)
            two_best_dec = []
            for p in range(Task_num):
                PopObj_acc = []
                PopObj_f1 = []
                PopObj_MCR = []
                PopObj_precision = []
                PopObj_sensitivity = []
                PopObj_specificity = []
                PopObj_auc = []
                PopObj_fit = []
                Population = Populations[p]
                Population_Dec = []
                for i in range(N):
                    fusion_para = Population.Decs[i][:]
                    fusion_feature_vectors = []
                    for j in range(len(Data[0])):
                        fusion_feature_vectors.append(np.concatenate((Data[0][j], Data[1][j]), axis=1) * fusion_para)
                    Population_Dec.append(fusion_feature_vectors)
                    acc, f1, MCR, precision, sensitivity, specificity, auc = CalObjs.cal_objects_test(fusion_feature_vectors, Population.Coding)
                    PopObj_acc.append(acc)
                    PopObj_f1.append(f1)
                    PopObj_MCR.append(MCR)
                    PopObj_precision.append(precision)
                    PopObj_sensitivity.append(sensitivity)
                    PopObj_specificity.append(specificity)
                    PopObj_auc.append(auc)
                    PopObj_fit.append(Population.Fits[i])
                #
                maxIdx = PopObj_fit.index(np.max(PopObj_fit))
                Acc = PopObj_acc[maxIdx]
                F1 = PopObj_f1[maxIdx]
                MCR = PopObj_MCR[maxIdx]
                precision = PopObj_precision[maxIdx]
                sensitivity = PopObj_sensitivity[maxIdx]
                specificity = PopObj_specificity[maxIdx]
                AUC = PopObj_auc[maxIdx]
                print('==============Test================')
                print('ACC:', Acc)
                print('F1:', F1)
                print('MCR:', MCR)
                print('precision:', precision)
                print('sensitivity:', sensitivity)
                print('specificity:', specificity)
                print('AUC:', AUC)
                #
                """
                    extract factors  
                """
                two_best_dec.append(Population_Dec[maxIdx])

            ef.extractFeatureFactors(two_best_dec, dataset, sel_labeled_ratio, "factor")
            end_time = time.time()
            print(f"Time cost: {end_time - start_time:.6f} s")
