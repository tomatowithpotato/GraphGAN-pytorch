import numpy as np
import pandas as pd
import re
import os


def str_list_to_float(str_list):
    return [float(item) for item in str_list]


def str_list_to_int(str_list):
    return [int(item) for item in str_list]


def read_edges(train_filename, test_filename=""):
    """read data from files
       for link_prediction and node_classify
    Args:
        train_filename: training file name
        test_filename: test file name
    Returns:
        node_num: int, number of nodes in the graph
        graph: dict, node_id -> list of neighbors in the graph
    """

    graph = {}
    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename) if test_filename != "" else []

    for edge in train_edges:
        if graph.get(edge[0]) is None:
            graph[edge[0]] = []
        if graph.get(edge[1]) is None:
            graph[edge[1]] = []
        graph[edge[0]].append(edge[1])
        graph[edge[1]].append(edge[0])

    for edge in test_edges:
        if graph.get(edge[0]) is None:
            graph[edge[0]] = []
        if graph.get(edge[1]) is None:
            graph[edge[1]] = []

    return graph

def read_edges_from_file(filename):
    with open(filename, "r") as f:
        lines = f.readlines()
        edges = [str_list_to_int(re.split('[\s,]+', line)[:-1]) for line in lines]
    return edges


def read_embeddings(filename, n_node, n_embed):
    """read pretrained node embeddings
    """

    embedding_matrix = np.random.rand(n_node, n_embed)
    try:
        with open(filename, "r") as f:
            lines = f.readlines()[1:]  # skip the first line
            for line in lines:
                emd = line.split()
                embedding_matrix[int(emd[0]), :] = str_list_to_float(emd[1:])
    except Exception as e:
        print("WARNNING: can not find the pre_embedding file")
    return embedding_matrix


def read_labels(filename, n_node):
    """read node labels
    """

    with open(filename, "r") as f:
        all_lines = f.readlines()
        n_classes, lines = int(all_lines[0].split()[0]), all_lines[1:]
        labels_matrix = np.zeros((n_node, n_classes))
        for line in lines:
            label = re.split('[\s,]+', line)[:-1]
            labels_matrix[int(label[0]), int(label[1])] = 1.0
    return n_classes, labels_matrix


def reindex_node_id(edges):
    """reindex the original node ID to [0, node_num)
    Args:
        edges: list, element is also a list like [node_id_1, node_id_2]
    Returns:
        new_edges: list[[1,2],[2,3]]
        new_nodes: list [1,2,3]
    """

    node_set = set()
    for edge in edges:
        node_set = node_set.union(set(edge))

    node_set = list(node_set)
    new_nodes = set()
    new_edges = []
    for edge in edges:
        new_edges.append([node_set.index(edge[0]), node_set.index(edge[1])])
        new_nodes = new_nodes.add(node_set.index(edge[0]))
        new_nodes = new_nodes.add(node_set.index(edge[1]))

    new_nodes = list(new_nodes)
    return new_edges, new_nodes


def generate_neg_links(train_filename, test_filename, test_neg_filename):
    """
    generate neg links for link prediction evaluation
    Args:
        train_filename: the training edges
        test_filename: the test edges
        test_neg_filename: the negative edges for test
    """

    train_edges = read_edges_from_file(train_filename)
    test_edges = read_edges_from_file(test_filename)
    neighbors = {}  # dict, node_ID -> list_of_neighbors
    for edge in train_edges + test_edges:
        if neighbors.get(edge[0]) is None:
            neighbors[edge[0]] = []
        if neighbors.get(edge[1]) is None:
            neighbors[edge[1]] = []
        neighbors[edge[0]].append(edge[1])
        neighbors[edge[1]].append(edge[0])
    nodes = set([x for x in range(len(neighbors))])

    # for each edge in the test set, sample a negative edge
    neg_edges = []

    for i in range(len(test_edges)):
        edge = test_edges[i]
        start_node = edge[0]
        neg_nodes = list(nodes.difference(set(neighbors[edge[0]] + [edge[0]])))
        neg_node = np.random.choice(neg_nodes, size=1)[0]
        neg_edges.append([start_node, neg_node])
    neg_edges_str = [str(x[0]) + "\t" + str(x[1]) + "\n" for x in neg_edges]
    with open(test_neg_filename, "w+") as f:
        f.writelines(neg_edges_str)


def softmax(x):
    e_x = np.exp(x - np.max(x))  # for computation stability
    return e_x / e_x.sum()

def make_config_dirs(config):
    for i in dir(config):
        if 'filename' in i:
            attribute = getattr(config, i)
            if isinstance(attribute, str):
                path = '/'.join(attribute.split('/')[:-1])
                if not os.path.exists(path):
                    os.makedirs(path)
            elif isinstance(attribute, list):
                for j in attribute:
                    path = '/'.join(j.split('/')[:-1])
                    if not os.path.exists(path):
                        os.makedirs(path)

if __name__ == "__main__":
    read_edges_rcmd("data/recommendation/MovieLens-1M/train.csv", 
                    "data/recommendation/MovieLens-1M/test.csv")