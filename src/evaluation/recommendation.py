import math
import numpy as np
import pandas as pd
import sys
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import f1_score
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.io import loadmat
from src import utils

class Recommendation:
    def __init__(self, embed_filename, n_node, n_embed, rcmd):
        self.embed_filename = embed_filename  # each line: node_id, embeddings(dim: n_embed)
        self.n_node = n_node
        self.n_embed = n_embed
        self.embed_filename = embed_filename

        self.watched = rcmd.watched # 测试集中用户看过的影片
        self.unwatched = rcmd.unwatched  # 训练集中用户未观看(评分)过的电影
        self.K = 1   

        self.emd = utils.read_embeddings(self.embed_filename, self.n_node, self.n_embed)
        self.scores = self.get_movie_score()        

    
    def eval_rcmd_K_movie(self, rcmd_K):       
        rcmd_K_movies = self.recommend_movie(self.scores, rcmd_K)
        precision, recall = self.get_precision_and_recall(rcmd_K_movies, self.watched)
        return {"precision": precision, "recall": recall}
    
    def get_movie_score(self):
        #预测每个用户对每部未观看电影的打分，得分最高的前K个电影即为推荐电影
        scores = {}
        for u in self.unwatched.keys():
            scores[u] = []
            for m in self.unwatched[u]:
                score = np.dot(self.emd[u], self.emd[m])
                scores[u].append((m, score))
            scores[u].sort(reverse=True, key=lambda x : x[1])
        
        return scores
    
    def recommend_movie(self, scores, rcmd_K):
        rcmd_K_movies = {}
        for u in scores:
            rcmd_K_movies[u] = set([score[0] for score in scores[u][:rcmd_K]])
        return rcmd_K_movies


    def get_precision_and_recall(self, rcmd_movies, watched_movies):
        u_set = rcmd_movies.keys()
        numerator = [len(rcmd_movies[u] & watched_movies[u]) for u in u_set]
        denominator_p = [min(len(rcmd_movies[u]), len(watched_movies[u])) for u in u_set]
        denominator_r = [len(watched_movies[u]) for u in u_set]

        precision = sum([a/b for a, b in zip(numerator, denominator_p)]) / len(u_set)
        recall = sum([a/b for a, b in zip(numerator, denominator_r)]) / len(u_set)
        return precision, recall