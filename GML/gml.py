import heapq
import time
from tqdm import tqdm
import numpy as np
from GmlMultiLabel.numbskull_extend import numbskull
import logging
from GmlMultiLabel import gml_utils
from GmlMultiLabel.evidential_support import EvidentialSupport
from GmlMultiLabel.evidence_select import EvidenceSelect
from GmlMultiLabel.approximate_probability_estimation import ApproximateProbabilityEstimation
from GmlMultiLabel.construct_subgraph import ConstructSubgraph
from configparser import ConfigParser
from concurrent.futures import ProcessPoolExecutor
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from os.path import join
from sklearn.metrics import confusion_matrix
from matplotlib import pyplot as plt
import torch
import torch.nn as nn
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import multilabel_confusion_matrix


class GML:
    """
     GML main process: evidential support->select_top-m->ApproximateProbabilityEstimation->select->top-k->inference->label->score
    """

    def __init__(self, variables, features, class_num, learning_method, top_m, top_k, top_n, update_proportion, balance,
                 optimization_threshold, learning_epoches, inference_epoches, nprocess, out):
        # check data
        self.labelcount = 0
        self.debug = True
        variables_keys = ['var_id', 'is_easy', 'is_evidence', 'true_label', 'label', 'feature_set']
        features_keys = ['feature_id', 'feature_type', 'parameterize', 'feature_name', 'weight']
        learning_methods = ['sgd', 'bgd']
        if learning_method not in learning_methods:
            raise ValueError('learning_methods has no this method: ' + learning_method)
        # check variables
        for variable in variables:
            for attribute in variables_keys:
                if attribute not in variable:
                    raise ValueError('variables has no key: ' + attribute)
        # check features
        for feature in features:
            for attribute in features_keys:
                if attribute not in feature:
                    raise ValueError('features has no key: ' + attribute)
        self.variables = variables
        self.features = features
        self.learning_method = learning_method  # now support sgd and bgd
        self.labeled_variables_set = set()
        self.top_m = top_m
        self.top_k = top_k
        self.top_n = top_n
        self.label1_count = 0
        self.classNum = class_num
        self.optimization_threshold = optimization_threshold  # entropy optimization threshold: less than 0 means no need to optimize
        self.update_proportion = update_proportion  # Evidence support update ratio: it is necessary to recalculate the evidence support after inferring a certain percentage of hidden variables
        self.select = EvidenceSelect(variables, features)
        self.subgraph = ConstructSubgraph(variables, features, class_num, balance)
        self.learing_epoches = learning_epoches  # Factor graph parameter learning rounds
        self.inference_epoches = inference_epoches  # Factor graph inference rounds
        self.nprocess = nprocess  # Number of multiple processes
        self.out = out  # Do you need to record the results
        self.evidence_interval_count = 10  # Number of evidence intervals divided by featureValue
        self.all_feature_set = set([x for x in range(0, len(features))])  # Feature ID collection
        self.observed_variables_set, self.poential_variables_set = gml_utils.separate_variables(
            variables)  # Dividing evidence variables and latent variables
        self.evidence_interval = gml_utils.init_evidence_interval(
            self.evidence_interval_count)  # Divide the evidence interval
        self.approximate = ApproximateProbabilityEstimation(variables, features, class_num)
        self.vname2pairid = {}
        self.support = EvidentialSupport(variables, features, class_num)
        gml_utils.init_bound(variables, features)  # Initial value of initialization parameter
        gml_utils.init_evidence(features, self.evidence_interval, self.observed_variables_set)
        # save results
        self.now = str(time.strftime('%Y-%m-%d-%H-%M-%S', time.localtime(time.time())))
        self.result = self.now + '-result.txt'
        if self.out:
            with open(self.result, 'w') as f:
                f.write(
                    'var_id' + ' ' + 'inferenced_probability' + ' ' + 'inferenced_label' + ' ' + 'ture_label' + '\n')
        # logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(name)s - [%(levelname)s]: %(message)s'
        )
        logging.info("GML inference begin")

    @staticmethod
    def initial(configFile, variables, features, class_num):
        '''
        load config from file
        @param configFile:
        @param variables:
        @param features:
        @return:
        '''
        config = ConfigParser()
        # Set default parameters
        config.read_dict({'para': {'learning_method': 'bgd',
                                   'learning_epoches': '1000',
                                   'inference_epoches': '1000',
                                   'top_m': '50',
                                   'top_k': '10',
                                   'top_n': '1',
                                   'n_process': '1',
                                   'update_proportion': '0.01',
                                   'balance': 'False',
                                   'optimization_threshold': '-1',
                                   'out': 'True'}
                          })
        config.read(configFile, encoding='UTF-8')
        learning_method = config['para']['learning_method']
        learning_epoches = int(config['para']['learning_epoches'])
        inference_epoches = int(config['para']['inference_epoches'])
        top_m = int(config['para']['top_m'])
        top_k = int(config['para']['top_k'])
        top_n = int(config['para']['top_n'])
        n_process = int(config['para']['n_process'])
        update_proportion = float(config['para']['update_proportion'])
        balance = config['para'].getboolean('balance')
        optimization_threshold = float(config['para']['optimization_threshold'])
        out = config['para'].getboolean('out')
        return GML(variables, features, class_num, learning_method, top_m, top_k, top_n, update_proportion, balance,
                   optimization_threshold, learning_epoches, inference_epoches, n_process, out)

    def evidential_support(self, poential_variables_set, update_feature_set):
        '''
        calculate evidential_support
        @param variable_set:
        @param update_feature_set:
        @return:
        '''
        for curc in range(0, self.classNum):
            self.support.evidential_support_by_regression(poential_variables_set, update_feature_set, c=curc)
        self.support.ev_by_regression(update_feature_set, c=-1)

    def approximate_probability_estimation(self, variable_set):
        '''
        estimation approximate_probability
        @param variable_set:
        @return:
        '''
        self.approximate.approximate_probability_estimation_by_interval(variable_set)

    def select_top_m_by_es(self, m):
        '''
        select tom m largest ES poential variables
        @param m:
        @return: m_id_list
        '''
        # If the current number of hidden variables is less than m, directly return to the hidden variable list
        if m > len(self.poential_variables_set):
            return list(self.poential_variables_set)
        poential_var_list = list()
        m_id_list = list()
        for var_id in self.poential_variables_set:
            if 'evidential_support' not in self.variables[var_id]:
                self.variables[var_id]['evidential_support'] = [.0] * self.classNum
            poential_var_list.append([var_id, self.variables[var_id]['evidential_support']])
        # Select the evidence to support th2e top m largest
        topm_var = heapq.nlargest(m, poential_var_list, key=lambda s: s[1])
        entropy_list = []
        for elem in topm_var:
            m_id_list.append(elem[0])
            entropy_list.append(self.variables[elem[0]]['entropy'])
        m_id_list = list(np.array(m_id_list)[np.argsort(np.array(entropy_list))])
        # logging.info('select m finished')
        return m_id_list

    def select_top_k_by_entropy(self, var_id_list, k):
        '''
        select top k  smallest entropy poential variables
        @param var_id_list:
        @param k:
        @return:
        '''
        # If the number of hidden variables is less than k, return the hidden variable list directly
        if len(var_id_list) < k:
            return var_id_list
        k_id_list = list(var_id_list[0:k])
        return k_id_list

    def evidence_select(self, var_id):
        '''
        Determine the subgraph structure
        @param var_id:
        @return:
        '''
        connected_var_set, connected_edge_set, connected_feature_set = self.select.select_evidence_by_multi(var_id)
        return connected_var_set, connected_edge_set, connected_feature_set

    def construct_subgraph(self, var_id):
        '''
        Construct subgraphs according to numbskull requirements
        @param var_id:
        @return:
        '''
        evidences = self.evidence_select(var_id)
        weight, variable, factor, fmap, domain_mask, edges_num, var_map, alpha_bound, tau_bound, weight_map_feature, sample_list, wmap, wfactor = self.subgraph.construct_subgraph_for_multi(
            var_id, evidences)
        return weight, variable, factor, fmap, domain_mask, edges_num, var_map, alpha_bound, tau_bound, weight_map_feature, sample_list, wmap, wfactor

    def inference_subgraph(self, var_id):
        '''
        Subgraph parameter learning and reasoning
        @param var_id:
        @return:
        '''
        if not type(var_id) == int:
            raise ValueError('var_id should be int')
        ns_learing = numbskull.NumbSkull(
            n_inference_epoch=self.learing_epoches,
            n_learning_epoch=self.inference_epoches,
            stepsize=0.01,
            decay=0.95,
            reg_param=1e-6,
            regularization=2,
            truncation=10,
            quiet=(not False),
            verbose=False,
            learn_non_evidence=True,
            sample_evidence=False,
            burn_in=20,
            nthreads=3,
            learning_method=self.learning_method
        )
        weight, variable, factor, fmap, domain_mask, edges_num, var_map, alpha_bound, tau_bound, weight_map_feature, sample_list, wmap, wfactor = self.construct_subgraph(
            var_id)
        subgraph = weight, variable, factor, fmap, domain_mask, edges_num, alpha_bound, tau_bound, sample_list, wmap, wfactor
        ns_learing.loadFactorGraph(*subgraph)
        # parameter learning
        ns_learing.learning()
        # logging.info("subgraph learning finished")

        # reasoning
        ns_inference = numbskull.NumbSkull(
            n_inference_epoch=self.learing_epoches,
            n_learning_epoch=self.inference_epoches,
            stepsize=0.001,
            decay=0.95,
            reg_param=1e-6,
            regularization=2,
            truncation=10,
            quiet=(not False),
            verbose=False,
            learn_non_evidence=False,
            sample_evidence=False,
            burn_in=10,
            nthreads=1,
            learning_method=self.learning_method
        )
        ns_inference.loadFactorGraph(*subgraph)
        ns_inference.inference()
        if type(var_id) == set or type(var_id) == list:
            if self.classNum == 2:
                for id in var_id:
                    self.variables[id]['inferenced_probability'] = ns_inference.factorGraphs[0].marginals[var_map[id]]
            else:
                inferenced_probability = ns_inference.factorGraphs[0].marginals.reshape((-1, self.classNum))  # 真推概率
                for id in var_id:
                    for c in range(self.classNum):
                        if 'inferenced_probability' not in self.variables[var_id].keys():
                            self.variables[var_id]['inferenced_probability'] = [0.0] * self.classNum
                        self.variables[id]['inferenced_probability'][c] = inferenced_probability[var_map[id]][c]
        elif type(var_id) == int:
            if self.classNum == 2:
                self.variables[var_id]['inferenced_probability'] = ns_inference.factorGraphs[0].marginals[
                    var_map[var_id]]
            else:
                inferenced_probability = ns_inference.factorGraphs[0].marginals.reshape((-1, self.classNum))
                if 'inferenced_probability' not in self.variables[var_id].keys():
                    self.variables[var_id]['inferenced_probability'] = [0.0] * self.classNum
                self.variables[var_id]['inferenced_probability'] = inferenced_probability[var_map[var_id]]
        return var_id, ns_inference.factorGraphs[0].marginals[var_map[var_id]]

    def label(self, var_id_list, isapprox, mlist=None):
        '''
        Select n from k inferred hidden variables for labeling
        @param var_id_list:
        @return:
        '''
        entropy_list = list()
        label_list = list()
        probability_indicator = None
        if isapprox == True:
            probability_indicator = 'approximate_probability'
        else:
            probability_indicator = 'inferenced_probability'
        # Calculate the entropy of k hidden variables
        if probability_indicator == 'inferenced_probability':
            for var_id in var_id_list:
                inferenced_probability_list = list()
                for c in range(self.classNum):
                    inferenced_probability_list.append(
                        gml_utils.entropy(self.variables[var_id]['inferenced_probability'][c]))
                self.variables[var_id]['entropy'] = min(inferenced_probability_list)
                entropy_list.append([var_id, self.variables[var_id]['entropy']])
        else:
            for var_id in var_id_list:
                var_index = var_id
                entropy_list.append([var_id, self.variables[var_index]['entropy']])
        # If labelnum is less than the number of variables passed in, mark top_n
        if len(var_id_list) >= self.top_n:
            var = list()
            min_var_list = heapq.nsmallest(self.top_n, entropy_list, key=lambda x: x[1])  # 选出熵最小的变量
            for mv in min_var_list:
                label_list.append(mv[0])
        # Otherwise mark all the variables passed in
        else:
            label_list = var_id_list
        for var_index in label_list:
            if mlist is not None:
                if var_index in mlist:
                    mlist.remove(var_index)
            self.variables[var_index]['probability'] = list(self.variables[var_index][probability_indicator])
            max_index = self.variables[var_index]['probability'].index(max(self.variables[var_index]['probability']))
            self.variables[var_index]['label'] = max_index
            self.poential_variables_set.remove(var_index)
            self.observed_variables_set.add(var_index)
            self.labeled_variables_set.add(var_index)
            self.variables[var_index]['is_evidence'] = True
            label_save = self.variables[var_index]['label']
            true_label = self.variables[var_index]['true_label']
        gml_utils.update_evidence(self.variables, self.features, label_list, self.evidence_interval)
        return len(label_list)

    def inference(self):
        '''
        Through the main process
        @return:
        '''
        labeled_var = 0
        labeled_count = 0
        update_feature_set = set()  # Stores features that have changed during a round of updates
        inferenced_variables_id = set()  # Hidden variables that have been established and inferred during a round of update
        pool = ProcessPoolExecutor(self.nprocess)
        if self.update_proportion > 0:  # 0.1
            update_cache = int(self.update_proportion * len(self.poential_variables_set))
        self.evidential_support(self.poential_variables_set, self.all_feature_set)
        self.approximate_probability_estimation(self.poential_variables_set)
        pbar = tqdm(total=len(self.poential_variables_set), desc="Inference", ncols=80)
        m_list = self.select_top_m_by_es(self.top_m)
        while len(self.poential_variables_set) > 0:
            labeled_count_begin = labeled_count
            if self.update_proportion > 0 and labeled_var >= update_cache:
                for var_id in self.labeled_variables_set:
                    for feature_id in self.variables[var_id]['feature_set'].keys():
                        update_feature_set.add(feature_id)
                # if len(update_feature_set) != 0:
                self.evidential_support(self.poential_variables_set, update_feature_set)
                self.approximate_probability_estimation(self.poential_variables_set)
                labeled_var = 0
                update_feature_set.clear()
                self.labeled_variables_set.clear()
                inferenced_variables_id.clear()
                m_list = self.select_top_m_by_es(self.top_m)
            elif len(m_list) == 0:
                m_list = self.select_top_m_by_es(self.top_m)
            k_list = self.select_top_k_by_entropy(m_list, self.top_k)
            approx_list = []
            add_list = [x for x in k_list if x not in inferenced_variables_id]
            for var_id in k_list:
                if self.optimization_threshold == -2 or (self.optimization_threshold >= 0 and self.variables[var_id][
                    'entropy'] <= self.optimization_threshold):
                    approx_list.append(var_id)
                # Added variables in each round of reasoning
            if len(approx_list) > 0:
                len_label_list = self.label(approx_list, isapprox=True, mlist=m_list)
                k_list.clear()
                labeled_var += len_label_list
                labeled_count += len_label_list
                pbar.update(len_label_list)
            else:
                for var_id in k_list:
                    if var_id in add_list:
                        var_id = int(var_id)
                        self.inference_subgraph(var_id)
                        # For the variables that have been inferred during each round of update, because the parameters are not updated, there is no need for inference.
                        inferenced_variables_id.add(var_id)
                len_label_list = self.label(k_list, isapprox=False, mlist=m_list)
                labeled_var += len_label_list
                labeled_count += len_label_list
                pbar.update(len_label_list)
        self.metric_multi()

    def align_cluster_probabilities(self, predLab, true_label):
        #
        cm = confusion_matrix(true_label, predLab)
        row_ind, col_ind = linear_sum_assignment(cm.max() - cm)
        label_mapping = dict(zip(col_ind, row_ind))
        aligned_labels = np.array([label_mapping[p] for p in predLab])
        return aligned_labels

    def metric_multi(self):
        pro = []
        for ii in range(len(self.variables)):
            if self.variables[ii]['is_easy'] == False:
                pro.append(self.variables[ii]['approximate_weight'])
        pro = np.array(pro)
        pro = np.exp(pro) / np.sum(np.exp(pro), axis=1)[:, np.newaxis]
        #
        m = nn.Softmax(dim=1)
        pro_cls_ = np.array(m(torch.from_numpy(pro)).cpu().numpy().tolist())
        correct = 0
        total = 0
        lab_pred = []
        lab_true = []
        for var in self.variables:
            if var['is_easy'] == False:
                total += 1
                if var['label'] == var['true_label']:
                    correct += 1
                lab_pred.append(var['label'])
                lab_true.append(var['true_label'])
        AUC = roc_auc_score(lab_true, pro_cls_, multi_class='ovr', average='macro')
        # AUC = roc_auc_score(lab_true, pro_cls_[:, 1])
        precision = precision_score(lab_true, lab_pred, average='macro')
        recall = recall_score(lab_true, lab_pred, average='macro')
        f1 = f1_score(lab_true, lab_pred, average='macro')
        #
        cm = confusion_matrix(lab_true, lab_pred)
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
        print()
        print('acc={}/{}={}'.format(correct, total, (correct / total)))
        print('precision={}'.format(precision))
        print('MCR={}'.format(mean_class_recall))
        print('f1={}'.format(f1))
        print('sensitivity={}'.format(sensitivity))
        print('specificity={}'.format(specificity))
        print('AUC={}'.format(AUC))
