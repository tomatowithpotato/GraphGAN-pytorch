import math
import numpy
import sys
from sklearn.multiclass import OneVsRestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from sklearn.metrics import f1_score
from sklearn.utils import shuffle as skshuffle
from sklearn.preprocessing import MultiLabelBinarizer
from scipy.io import loadmat
from src import utils


class TopKRanker(OneVsRestClassifier):
    def predict(self, X, top_k_list):
        assert X.shape[0] == len(top_k_list)
        probs = numpy.asarray(super(TopKRanker, self).predict_proba(X))
        all_labels = []
        for i, k in enumerate(top_k_list):
            probs_ = probs[i, :]
            labels = self.classes_[probs_.argsort()[-k:]].tolist()
            all_labels.append(labels)
        return all_labels

class NodeClassificationEval():
    def __init__(self, embed_filename, labels_matrix, n_node, n_embed, n_classes):
        self.embed_filename = embed_filename  # each line: node_id, embeddings(dim: n_embed)
        self.n_node = n_node
        self.n_embed = n_embed
        self.n_classes = n_classes
        self.emd = utils.read_embeddings(embed_filename, n_node=n_node, n_embed=n_embed)
        self.labels_matrix = labels_matrix
    
    def eval_node_classification(self): 
        labels_count = self.labels_matrix.shape[1]
        mlb = MultiLabelBinarizer(classes=list(range(labels_count)))

        x_train, y_train = self.emd[ : int(0.9*self.n_node), :], self.labels_matrix[ : int(0.9*self.n_node), :]
        x_test, y_test = self.emd[int(0.9*self.n_node) : , :], self.labels_matrix[int(0.9*self.n_node) : , :]

        clf = TopKRanker(LogisticRegression())
        clf.fit(x_train, y_train)

        y_test_coo = [[] for _ in range(y_test.shape[0])]
        for i in range(y_test.shape[0]):
            for j, e in enumerate(y_test[i, :]):
                if math.isclose(e, 1.0):
                    y_test_coo[i].append(j)
        
        top_k_list = [len(l) for l in y_test_coo]
        a, b = max(top_k_list), min(top_k_list)
        preds = clf.predict(x_test, top_k_list)

        results = {}
        averages = ["micro", "macro"]
        results["acc"] = accuracy_score(mlb.fit_transform(y_test_coo), mlb.fit_transform(preds))
        for average in averages:
            results[average] = f1_score(mlb.fit_transform(y_test_coo), mlb.fit_transform(preds), average=average)
        
        return results


if __name__ == "__main__":
    matfile = 'blogcatalog.mat'
    mat = loadmat(matfile)
    labels_matrix = mat['group']
    print(labels_matrix.shape)
    print(labels_matrix[5, 3])