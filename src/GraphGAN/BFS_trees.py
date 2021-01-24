import os
import sys
import tqdm
import pickle
import collections
import config

class BFS_trees:
    '''可分批读取数据的BFS树，主要是为了避免占用太多内存
       如果你内存够用，建议把config.py里的cache_batch的里的值都设为1，即不分批
    '''
    def __init__(self, root_nodes, graph, batch_num=1, app=None, user_count=None):
        self.app = app
        self.root_nodes = root_nodes
        self.graph = graph
        self.batch_num = batch_num
        self.index = -1
        self.step = int(len(root_nodes) / self.batch_num)

        if self.app == "recommendation":
            self.user_count = user_count
        
        self.BFS_trees = None
        
    
    def read_from_file(self, filename, present_nodes):
        BFS_trees = None
        if os.path.isfile(filename):
            print("reading BFS-trees from cache ... " + str(self.index))
            pickle_file = open(filename, 'rb')
            BFS_trees = pickle.load(pickle_file)
            pickle_file.close()
        else:
            print("constructiong BFS-trees " + str(self.index))

            if self.app == "recommendation":
                BFS_trees = self.construct_trees_rcmd(present_nodes)
            else:
                BFS_trees = self.construct_trees(present_nodes)

            pickle_file = open(filename, 'wb')
            pickle.dump(BFS_trees, pickle_file)
            pickle_file.close()
        return BFS_trees

    def get_tree(self, root):
        if self.index != int(root / self.step):
            self.index = int(root / self.step)
            filename = config.cache_filename + '_' + str(self.index)
            r_bound = min(self.step*(self.index+1), len(self.root_nodes))
            present_nodes = self.root_nodes[self.step*self.index : r_bound]
            self.BFS_trees = self.read_from_file(filename, present_nodes)
        return self.BFS_trees[root]
    

    def construct_trees(self, nodes):
        """use BFS algorithm to construct the BFS-trees
        Args:
            nodes: the list of nodes in the graph
        Returns:
            trees: dict, root_node_id -> tree, where tree is a dict: node_id -> list: [father, child_0, child_1, ...]
        """
        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            trees[root][root] = [root]
            used_nodes = set()
            queue = collections.deque([root])
            while len(queue) > 0:
                cur_node = queue.popleft()
                used_nodes.add(cur_node)
                for sub_node in self.graph[cur_node]:
                    if sub_node not in used_nodes:
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
                        queue.append(sub_node)
                        used_nodes.add(sub_node)
        return trees
    

    def construct_trees_rcmd(self, nodes):
        """for recommendation
        \n除了root节点为用户外，路径上的所有节点都是电影节点
        """
        trees = {}
        for root in tqdm.tqdm(nodes):
            trees[root] = {}
            trees[root][root] = [root]
            used_nodes = set()
            queue = collections.deque([root])
            while len(queue) > 0:
                cur_node = queue.popleft()
                used_nodes.add(cur_node)
                for sub_node in self.graph[cur_node]:
                    # 意味着该节点是用户节点，故应当去链接其所连的其他电影节点
                    if sub_node < self.user_count:
                        for m_node in self.graph[sub_node]:
                            if m_node not in used_nodes:
                                trees[root][cur_node].append(m_node)
                                trees[root][m_node] = [cur_node]
                                queue.append(m_node)
                                used_nodes.add(m_node)
                    elif sub_node not in used_nodes:
                        trees[root][cur_node].append(sub_node)
                        trees[root][sub_node] = [cur_node]
                        queue.append(sub_node)
                        used_nodes.add(sub_node)
        return trees