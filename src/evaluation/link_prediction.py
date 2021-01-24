"""
The class is used to evaluate the application of link prediction
"""

import numpy as np
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from src import utils


class LinkPredictEval(object):
    def __init__(self, embed_filename, test_filename, test_neg_filename, n_node, n_embed):
        self.embed_filename = embed_filename  # each line: node_id, embeddings(dim: n_embed)
        self.test_filename = test_filename  # each line: node_id1, node_id2
        self.test_neg_filename = test_neg_filename  # each line: node_id1, node_id2
        self.n_node = n_node
        self.n_embed = n_embed
        self.emd = utils.read_embeddings(embed_filename, n_node=n_node, n_embed=n_embed)

    def eval_link_prediction(self, emd=None):
        if emd is not None:
            self.emd = emd

        test_edges = utils.read_edges_from_file(self.test_filename)
        test_edges_neg = utils.read_edges_from_file(self.test_neg_filename)
        
        test_edges.extend(test_edges_neg)  #test_edges前半部分边均为正样本，后半部分均为负样本

        # may exists isolated point
        score_res = []
        for i in range(len(test_edges)):
            score_res.append(np.dot(self.emd[test_edges[i][0]], self.emd[test_edges[i][1]]))
        test_label = np.array(score_res)
        
        median = np.median(test_label)  #这行代码按得分中位数划分预测结果，tp+fp==fn+tn
        index_pos = test_label >= median
        index_neg = test_label < median
        test_label[index_pos] = 1
        test_label[index_neg] = 0
        true_label = np.zeros(test_label.shape)
       
        true_label[0: len(true_label) // 2] = 1  #前半部分边均为正标签，后半部分均为负标签，tp+fn==fp+tn

        '''
        由 tp+fp==fn+tn, tp+fn==fp+tn => tp==tn, fn==fp
        因此accuracy, precision, recall三者相等，故accuracy等于f1-macro
        '''
        #test_eval(true_label, test_label)

        accuracy = accuracy_score(true_label, test_label)
        macro = f1_score(true_label, test_label, average="macro")

        return {"acc": accuracy, "macro": macro}

def test_eval(true_label, test_label):
    '''测试评估是否正确'''
    accuracy = accuracy_score(true_label, test_label)
    macro = f1_score(true_label, test_label, average="macro")

    recall = recall_score(true_label, test_label)
    precision = precision_score(true_label, test_label)
    print('acc: {}, precision: {}, recall: {}'.format(accuracy, precision, recall))

    '''
    tp 实际真，预测真 1,1
    fp 实际假，预测真 0,1
    fn 实际真，预测假 1,0
    tn 实际假，预测假 0,0
    '''
    tp = len(np.argwhere(true_label + test_label == 2))
    fn = len(np.argwhere(true_label - test_label == -1))
    fp = len(np.argwhere(true_label - test_label == 1))
    tn = len(np.argwhere(true_label + test_label == 0))
    print('tp: {}, fn: {}, fp: {}, tn: {}'.format(tp, fn, fp, tn))
    '''
    accuracy, precision, recall三者相等，当且仅当tp==tn，fn==fp
    '''
    print((tp+tn)/(tp+tn+fp+fn), tp/(tp+fp), tp/(tp+fn))
