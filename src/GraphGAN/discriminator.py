import torch
import torch.nn as nn
import config

class Discriminator(nn.Module):
    def __init__(self, n_node, node_emd_init):
        super(Discriminator, self).__init__()
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        self.embedding_matrix = nn.Parameter(torch.tensor(node_emd_init))
        self.bias = nn.Parameter(torch.zeros([self.n_node]))

        self.node_embedding = None
        self.node_neighbor_embedding = None
        self.neighboor_bias = None
    
    def score(self, node_id, node_neighbor_id):
        '''
        score: 一个n维向量，n为节点数。假设用向量d_v表示点v, 则d_v与各样本点v1的表示向量d_v1的内积组成的向量即为score
                表示向量的内积涵义为两节点相连的评分
        '''
        self.node_embedding = self.embedding_matrix[node_id, :]
        self.node_neighbor_embedding = self.embedding_matrix[node_neighbor_id, :]
        self.neighboor_bias = self.bias[node_neighbor_id]
        #print(torch.sum(input=self.node_embedding * self.node_neighbor_embedding, dim=1).shape)
        #print(self.bias[node_neighbor_id].shape)
        return torch.sum(input=self.node_embedding * self.node_neighbor_embedding, dim=1) + self.neighboor_bias
    
    def loss(self, node_id, node_neighbor_id, label):
        '''
        我们的目标是 maximize mean(log(D(x)) + log(1 - D(G(z))))

        因为BCEloss的结果为 -mean(log(D(x)) + log(1 - D(G(z))))

        所以直接对BCEloss的结果梯度下降即可
        '''
        l2_loss = lambda x: torch.sum(x * x) / 2 * config.lambda_dis
        prob = self.forward(self.score(node_id, node_neighbor_id))     
        criterion = nn.BCELoss()
        #正则项
        regularization = l2_loss(self.node_embedding) + l2_loss(self.node_neighbor_embedding) + l2_loss(self.neighboor_bias) 
        _loss = criterion(prob, torch.tensor(label).double()) + regularization
        return _loss
    
    def reward(self, node_id, node_neighbor_id):
        '''
        强化学习，用于generator的训练
        '''
        return torch.log(1 + torch.exp(self.score(node_id, node_neighbor_id))).detach()
    
    def forward(self, score):
        return torch.sigmoid(score)


if __name__ == "__main__":
    import numpy as np
    import sys, os
    sys.path.append("../..")
    os.chdir(sys.path[0]) 
    from src import utils
    
    n_node, graph = utils.read_edges(train_filename=config.train_filename, test_filename=config.test_filename)
    node_embed_init_d = utils.read_embeddings(filename=config.pretrain_emb_filename_d,
                                              n_node=n_node,
                                              n_embed=config.n_emb)
    discriminator = Discriminator(n_node=n_node, node_emd_init=node_embed_init_d)
    print(discriminator.loss(np.array([1,2,3,4]), np.array([5,6,7,8]), np.array([0,1,0,1])))
    
    print(type(discriminator.state_dict()))
    print(discriminator.state_dict())
    print("-"*80)
    for p in discriminator.parameters():
        print(p)
    