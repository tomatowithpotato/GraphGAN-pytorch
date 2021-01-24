import os
import re
import random
import pandas as pd

def create_test_neg():
    all_filepath = 'data/node_classification/BlogCatalog.txt'
    test_filepath = 'data/node_classification/BlogCatalog/test.txt'
    test_neg_filepath = 'data/node_classification/BlogCatalog/test_neg.txt'
    pos_edges = {}
    all_nodes = set()
    with open(all_filepath, 'r') as f:
        lines = f.readlines()
        lst = [re.split('[\s,]+', line)[:-1] for line in lines]
        for item in lst:
            all_nodes.add(item[0])
            all_nodes.add(item[1])
            if item[0] in pos_edges:
                pos_edges[item[0]].add(item[1])
            else:
                pos_edges[item[0]] = set()

    with open(test_filepath, 'r') as test_f:
        lines = test_f.readlines()
        lst = [re.split('[\s,]+', line)[:-1] for line in lines]
        node1 = [item[0] for item in lst]

    all_nodes = list(all_nodes)
    neg_edges = []
    
    for node in node1:
        neg_node = random.choice(all_nodes)
        while neg_node in pos_edges[node]:
            neg_node = random.choice(all_nodes)
        neg_edges.append(','.join([node, neg_node]) + '\n')
    
    with open(test_neg_filepath, 'w') as test_neg_f:
        test_neg_f.writelines(neg_edges)


def create_neg():
    train_filepath = 'data/node_classification/BlogCatalog/train.txt'
    neg_filepath = 'data/node_classification/BlogCatalog/neg.txt'
    pos_edges = {}
    all_nodes = set()
    node1 = []
    with open(train_filepath, 'r') as f:
        lines = f.readlines()
        lst = [re.split('[\s,]+', line)[:-1] for line in lines]
        for item in lst:
            node1.append(item[0])
            all_nodes.add(item[0])
            all_nodes.add(item[1])
            if item[0] in pos_edges:
                pos_edges[item[0]].add(item[1])
            else:
                pos_edges[item[0]] = set()
    
    all_nodes = list(all_nodes)
    neg_edges = []
    
    for node in node1:
        neg_node = random.choice(all_nodes)
        while neg_node in pos_edges[node]:
            neg_node = random.choice(all_nodes)
        neg_edges.append(','.join([node, neg_node]) + '\n')
    
    with open(neg_filepath, 'w') as test_neg_f:
        test_neg_f.writelines(neg_edges)

def split_train_test():
    all_filepath = 'data/node_classification/BlogCatalog.txt'
    train_filepath = 'data/node_classification/BlogCatalog/train.txt'
    test_filepath = 'data/node_classification/BlogCatalog/test.txt'

    with open(all_filepath, 'r') as f:
        lines = f.readlines()
        all_len = len(lines)
        ratio = 0.9
        train_size = int(ratio*all_len)
        random.shuffle(lines)
        with open(train_filepath, 'w') as train_f:
            train_f.writelines(lines[:train_size])
        with open(test_filepath, 'w') as test_f:
            test_f.writelines(lines[train_size:])

def create_test_neg_csv():
    train_filepath = 'data/recommendation/MovieLens-1M/train.csv'
    test_filepath = 'data/recommendation/MovieLens-1M/test.csv'
    test_neg_filepath = 'data/recommendation/MovieLens-1M/test_neg.csv'
    train_set = pd.read_csv(train_filepath)
    test_set = pd.read_csv(test_filepath)
    test_u, all_m = [], []
    pos = {}
    #print(train_set.loc[0, 'user_id'], train_set.loc[0, 'movie_id'])
    
    data_set = [train_set, test_set]
    for index, ds in enumerate(data_set):
        u_lst, m_lst = ds['user_id'].values.tolist(), ds['movie_id'].values.tolist()
        if index == 1:
            test_u.extend(u_lst)
        all_m.extend(m_lst)
        
        for u, m in zip(u_lst, m_lst):
            if u not in pos:
                pos[u] = set()
            pos[u].add(m)

    neg_list = []
    for u in test_u:
        m = random.choice(all_m)
        while m in pos[u]:
            m = random.choice(all_m)
        neg_list.append(m)
    
    test_neg_set = test_set
    test_neg_set.loc[:, 'movie_id'] = neg_list
    test_neg_set.to_csv(test_neg_filepath, index=False)


def split_train_test_csv():
    all_filepath = "data/recommendation/MovieLens-1M/ratings.csv"
    train_filepath = 'data/recommendation/MovieLens-1M/train.csv'
    test_filepath = 'data/recommendation/MovieLens-1M/test.csv'

    all_file = pd.read_csv(all_filepath)
    train_set = all_file.sample(frac=0.9, random_state=0, axis=0)
    test_set = all_file[~all_file.index.isin(train_set.index)]

    train_set.to_csv(train_filepath, index=False)
    test_set.to_csv(test_filepath, index=False)


def modify_index():
    train_filepath = 'data/node_classification/BlogCatalog/train.txt'
    neg_filepath = 'data/node_classification/BlogCatalog/neg.txt'
    labels_filepath = 'data/node_classification/BlogCatalog/labels.txt'
    for filepath in [train_filepath, neg_filepath]:
        print(filepath)
        with open(filepath, 'r') as f:
            lines = f.readlines()
            content = [re.split('[\s,]+', line)[:-1] for line in lines]
            content = [','.join([str(int(c[0])-1), str(int(c[1])-1)]) + '\n' for c in content]
            a = 1
        with open(filepath, 'w') as f:
            f.writelines(content)

    with open(labels_filepath, 'r') as f:
        lines = f.readlines()
        n_classes, lines = lines[0], lines[1:]
        content = [re.split('[\s,]+', line)[:-1] for line in lines]
        content = [','.join([str(int(c[0])-1), str(int(c[1])-1)]) + '\n' for c in content]
        content = [n_classes] + content
    with open(labels_filepath, 'w') as f:
        f.writelines(content)

def select_4_5_star():
    ratings_file = "data/recommendation/MovieLens-1M/ratings.csv"
    all_ratings_file = "data/recommendation/MovieLens-1M dataset/ratings.csv"
    all_ratings = pd.read_csv(all_ratings_file)
    ratings = all_ratings[all_ratings['rating'] >= 4]
    
    print(all_ratings)
    print(ratings)

    ratings.to_csv(ratings_file, index=False)

if __name__ == "__main__":
    # ratings = pd.read_csv('data/recommendation/MovieLens-1M/test.csv')
    # user_list = ratings["user_id"].values.tolist()
    # movie_list = ratings["movie_id"].values.tolist()
    # rating_list = ratings["rating"].values.tolist()
    # #print(type(user_list), type(movie_list), type(rating_list))
    # all_list = list(zip(user_list, movie_list, rating_list))
    # print(len(all_list))
    create_test_neg_csv()