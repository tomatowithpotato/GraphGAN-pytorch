import torch
import torch.nn as nn
import config


class Generator(nn.Module):
    def __init__(self, n_node, node_emd_init):
        super(Generator, self).__init__()
        self.n_node = n_node
        self.node_emd_init = node_emd_init

        self.embedding_matrix = nn.Parameter(torch.tensor(node_emd_init))
        self.bias = nn.Parameter(torch.zeros([self.n_node]))
        
        self.node_embedding = None
        self.node_neighbor_embedding = None
    
    def all_score(self):
        return torch.matmul(self.embedding_matrix, torch.transpose(self.embedding_matrix, 0, 1)).detach()
    
    def score(self, node_id, node_neighbor_id):
        '''
        score: 一个n维向量，n为节点数。假设用向量g_v表示点v, 则g_v与各样本点v1的表示向量g_v1的内积组成的向量即为score
        '''
        self.node_embedding = self.embedding_matrix[node_id, :]
        self.node_neighbor_embedding = self.embedding_matrix[node_neighbor_id, :]
        return torch.sum(input=self.node_embedding * self.node_neighbor_embedding, dim=1) + self.bias[node_neighbor_id]
    
    def loss(self, prob, reward):
        '''
        Args:
            prob: D(Z)
            reward: 强化学习的奖励因子

        原始的生成器损失函数为 minimize mean(log(1-D(Z))), Z为负样本

        但是原始的损失函数无法提供足够梯度，导致生成器得不到训练

        作为替代，实际运行时使用的是 maximize mean(log(D(Z)))

        因此，对 -mean(log(D(Z))) 梯度下降即可
        '''
        l2_loss = lambda x: torch.sum(x * x) / 2 * config.lambda_gen
        prob = torch.clamp(input=prob, min=1e-5, max=1)
        #正则项
        regularization = l2_loss(self.node_embedding) + l2_loss(self.node_neighbor_embedding)
        _loss = -torch.mean(torch.log(prob) * reward) + regularization
        
        return _loss


if __name__ == "__main__":
    import sys, os
    sys.path.append("../..")
    os.chdir(sys.path[0]) 
    from src import utils
    n_node, graph = utils.read_edges(train_filename=config.train_filename, test_filename=config.test_filename)
    node_embed_init_g = utils.read_embeddings(filename=config.pretrain_emb_filename_g,
                                              n_node=n_node,
                                              n_embed=config.n_emb)
    generator = Generator(n_node=n_node, node_emd_init=node_embed_init_g)
    for p in generator.parameters():
        print(p.name)
        print(p)
    