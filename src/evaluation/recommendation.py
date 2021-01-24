import math
import numpy as np
import sys
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.io import loadmat
from src import utils

class Recommendation:
    def __init__(self, embed_filename, test_filename, test_neg_filename, 
                 n_node, n_embed, user_max):
        self.embed_filename = embed_filename  # each line: node_id, embeddings(dim: n_embed)
        self.test_filename = test_filename  # each line: node_id1, node_id2
        self.test_neg_filename = test_neg_filename  # each line: node_id1, node_id2
        self.n_node = n_node
        self.n_embed = n_embed
        self.emd = utils.read_embeddings(embed_filename, n_node=n_node, n_embed=n_embed)
        self.user_max = user_max
    
    def eval_rcmd_lp(self, emd=None):
        if emd is not None:
            self.emd = emd

        test_edges = utils.read_edges_from_file_rcmd(self.test_filename)
        test_edges_neg = utils.read_edges_from_file_rcmd(self.test_neg_filename)
        
        test_edges.extend(test_edges_neg)  #test_edges前半部分边均为正样本，后半部分均为负样本

        # may exists isolated point
        score_res = []
        for i in range(len(test_edges)):
            u_id, m_id = test_edges[i][0]-1, test_edges[i][1]-1+self.user_max
            score_res.append(np.dot(self.emd[u_id], self.emd[m_id]))
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
    
    def eval_rcmd_K(self, unwatched_movies, emd=None):
        if emd is not None:
            self.emd = emd
        pass