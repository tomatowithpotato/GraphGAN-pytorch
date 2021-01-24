import torch
import torch.nn as nn
import os
import sys
import tqdm
import pickle
import numpy as np
import collections
from discriminator import Discriminator
from generator import Generator
import config
from src import utils
from src.evaluation import link_prediction as lp
from src.evaluation import node_classification as nc
from BFS_trees import BFS_trees


class graphGAN():
    def __init__(self):
        utils.make_config_dirs(config)

        #read graph
        if config.app == "link_prediction":
            self.graph = utils.read_edges(train_filename=config.train_filename, test_filename=config.test_filename)
            self.n_node = max(list(self.graph.keys())) + 1
        elif config.app == "node_classification":
            self.graph = utils.read_edges(train_filename=config.train_filename)
            self.n_node = max(list(self.graph.keys())) + 1
            self.n_classes ,self.labels_matrix = utils.read_labels(filename=config.labels_filename, n_node=self.n_node)
        elif config.app == "recommendation":
            self.graph, self.user_max, self.unwatched = utils.read_edges_rcmd(train_filename=config.rcmd_train_filename, 
                                                                              test_filename=config.rcmd_test_filename)
            self.n_node = max(list(self.graph.keys())) + 1                                       
        else:
            raise Exception("Unknown task: {}".format(config.app))
        
        #read root nodes
        if config.app == "recommendation":
            self.root_nodes = sorted(list(self.graph.keys()))[:self.user_max]
        else:
            self.root_nodes = sorted(list(self.graph.keys()))

        #read pre_emb matrix
        node_embed_init_d = utils.read_embeddings(filename=config.pretrain_emb_filename_d,
                                                       n_node=self.n_node,
                                                       n_embed=config.n_emb)
        node_embed_init_g = utils.read_embeddings(filename=config.pretrain_emb_filename_g,
                                                       n_node=self.n_node,
                                                       n_embed=config.n_emb)
        self.discriminator = Discriminator(n_node=self.n_node, node_emd_init=node_embed_init_d)
        self.generator = Generator(n_node=self.n_node, node_emd_init=node_embed_init_g)
        
        #construct BFS-tree
        if config.app == "recommendation":
            self.BFS_trees = BFS_trees(self.root_nodes, self.graph, batch_num=config.cache_batch, 
                                       app=config.app, user_max=self.user_max)
        else:
            self.BFS_trees = BFS_trees(self.root_nodes, self.graph, batch_num=config.cache_batch)
        
        

    def prepare_data_for_d(self):
        """generate positive and negative samples for the discriminator, and record them in the txt file"""
        print("prepare_data_for_d")
        center_nodes = []
        neighbor_nodes = []
        labels = []
        for i in tqdm.tqdm(self.root_nodes):
            if np.random.rand() < config.update_ratio:
                pos = self.graph[i]
                neg, _ = self.sample(i, self.BFS_trees.get_tree(i), len(pos), for_d=True)
                if len(pos) != 0 and neg is not None:
                    # positive samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(pos)
                    labels.extend([1] * len(pos))

                    # negative samples
                    center_nodes.extend([i] * len(pos))
                    neighbor_nodes.extend(neg)
                    labels.extend([0] * len(neg))
        return center_nodes, neighbor_nodes, labels


    def prepare_data_for_g(self):
        """sample nodes for the generator"""
        print("prepare_data_for_g")
        paths = []
        for i in tqdm.tqdm(self.root_nodes):
            if np.random.rand() < config.update_ratio:
                sample, paths_from_i = self.sample(i, self.BFS_trees.get_tree(i), config.n_sample_gen, for_d=False)
                if paths_from_i is not None:
                    paths.extend(paths_from_i)
        node_pairs = list(map(self.get_node_pairs_from_path, paths))
        node_1 = []
        node_2 = []
        for i in range(len(node_pairs)):
            for pair in node_pairs[i]:
                node_1.append(pair[0])
                node_2.append(pair[1])

        reward = self.discriminator.reward(node_1, node_2)
        return node_1, node_2, reward
    

    def sample(self, root, tree, sample_num, for_d):
        """ sample nodes from BFS-tree
        Args:
            root: int, root node
            tree: dict, BFS-tree
            sample_num: the number of required samples
            for_d: bool, whether the samples are used for the generator or the discriminator
        Returns:
            samples: list, the indices of the sampled nodes
            paths: list, paths from the root to the sampled nodes
        """

        all_score = self.generator.all_score().numpy()
        samples = []
        paths = []
        n = 0

        while len(samples) < sample_num:
            current_node = root
            previous_node = -1
            paths.append([])
            is_root = True
            paths[n].append(current_node)
            while True:
                node_neighbor = tree[current_node][1:] if is_root else tree[current_node]
                is_root = False
                if len(node_neighbor) == 0:  # the tree only has a root
                    return None, None
                if for_d:  # skip 1-hop nodes (positive samples)
                    if node_neighbor == [root]:
                        # in current version, None is returned for simplicity
                        return None, None
                    if root in node_neighbor:
                        node_neighbor.remove(root)
                relevance_probability = all_score[current_node, node_neighbor]
                relevance_probability = utils.softmax(relevance_probability)
                next_node = np.random.choice(node_neighbor, size=1, p=relevance_probability)[0]  # select next node
                paths[n].append(next_node)
                if next_node == previous_node:  # terminating condition
                    samples.append(current_node)
                    break
                previous_node = current_node
                current_node = next_node
            n = n + 1
        return samples, paths


    @staticmethod
    def get_node_pairs_from_path(path):
        """
        given a path from root to a sampled node, generate all the node pairs within the given windows size
        e.g., path = [1, 0, 2, 4, 2], window_size = 2 -->
        node pairs= [[1, 0], [1, 2], [0, 1], [0, 2], [0, 4], [2, 1], [2, 0], [2, 4], [4, 0], [4, 2]]
        :param path: a path from root to the sampled node
        :return pairs: a list of node pairs
        """

        path = path[:-1]
        pairs = []
        for i in range(len(path)):
            center_node = path[i]
            for j in range(max(i - config.window_size, 0), min(i + config.window_size + 1, len(path))):
                if i == j:
                    continue
                node = path[j]
                pairs.append([center_node, node])
        return pairs
    

    def write_embeddings_to_file(self):
        """write embeddings of the generator and the discriminator to files"""

        modes = [self.generator, self.discriminator]
        for i in range(2):
            embedding_matrix = modes[i].embedding_matrix.detach().numpy()
            index = np.array(range(self.n_node)).reshape(-1, 1)
            embedding_matrix = np.hstack([index, embedding_matrix])
            embedding_list = embedding_matrix.tolist()
            embedding_str = [str(int(emb[0])) + "\t" + "\t".join([str(x) for x in emb[1:]]) + "\n"
                             for emb in embedding_list]
            with open(config.emb_filenames[i], "w+") as f:
                lines = [str(self.n_node) + "\t" + str(config.n_emb) + "\n"] + embedding_str
                f.writelines(lines)
    