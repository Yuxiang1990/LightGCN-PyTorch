"""
Created on Mar 1, 2020
Pytorch Implementation of LightGCN in
Xiangnan He et al. LightGCN: Simplifying and Powering Graph Convolution Network for Recommendation

@author: Jianbai Ye (gusye@mail.ustc.edu.cn)

Define models here
"""
import world
import torch
from dataloader import BasicDataset
from torch import nn
import numpy as np

class BasicModel(nn.Module):    
    def __init__(self):
        super(BasicModel, self).__init__()
    
    def getUsersRating(self, users):
        raise NotImplementedError
    
class PairWiseModel(BasicModel):
    def __init__(self):
        super(PairWiseModel, self).__init__()
    def bpr_loss(self, users, pos, neg):
        """
        Parameters:
            users: users list 
            pos: positive items for corresponding users
            neg: negative items for corresponding users
        Return:
            (log-loss, l2-loss)
        """
        raise NotImplementedError


class LightGCN_svd(BasicModel):
    def __init__(self, 
                 config:dict, 
                 dataset:BasicDataset):
        super(LightGCN_svd, self).__init__()
        self.config = config
        self.dataset : LightGCN_svd.BasicDataset = dataset
        self.beta = 2.0
        self.coef_u = 0
        self.coef_i = 0
        self.req_vec = 400
        self.__init_weight()


    def weight_func(self, sig):
        return torch.exp(self.beta * sig)

    def __init_weight(self):
        self.num_users  = self.dataset.n_users
        self.num_items  = self.dataset.m_items
        self.latent_dim = self.config['latent_dim_rec']
        self.n_layers = self.config['lightGCN_n_layers']
        self.keep_prob = self.config['keep_prob']
        self.A_split = self.config['A_split']
        self.embedding_user = torch.nn.Embedding(
            num_embeddings=self.num_users, embedding_dim=self.latent_dim)
        self.embedding_item = torch.nn.Embedding(
            num_embeddings=self.num_items, embedding_dim=self.latent_dim)
        if self.config['pretrain'] == 0:
#             nn.init.xavier_uniform_(self.embedding_user.weight, gain=1)
#             nn.init.xavier_uniform_(self.embedding_item.weight, gain=1)
#             print('use xavier initilizer')
# random normal init seems to be a better choice when lightGCN actually don't use any non-linear activation function
            nn.init.normal_(self.embedding_user.weight, std=0.1)
            nn.init.normal_(self.embedding_item.weight, std=0.1)
            world.cprint('use NORMAL distribution initilizer')
        else:
            self.embedding_user.weight.data.copy_(torch.from_numpy(self.config['user_emb']))
            self.embedding_item.weight.data.copy_(torch.from_numpy(self.config['item_emb']))
            print('use pretarined data')
        self.f = nn.Sigmoid()
        *_, self.norm_adj = self.dataset.getSparseGraph()
        # rate_matrix = self.norm_adj[:self.num_users, self.num_users:].todense()
        # self.rate_matrix = torch.from_numpy(rate_matrix).cuda()
        # U, value, V = torch.svd_lowrank(self.rate_matrix, q=400, niter=30)
        # np.save('./svd_u.npy', U.cpu().numpy())
        # np.save('./svd_v.npy', V.cpu().numpy())
        # np.save('./svd_value.npy', value.cpu().numpy())
        self.rate_matrix = torch.Tensor(np.load('/mnt/lustre/yeyuxiang/projects/LightGCN-PyTorch/svd/svd_gcn/datasets/gowalla/rate_sparse.npy')).cuda()
        value = torch.Tensor(np.load('/mnt/lustre/yeyuxiang/projects/LightGCN-PyTorch/svd/svd_gcn/datasets/gowalla/svd_value.npy')).cuda()
        U = torch.Tensor(np.load('/mnt/lustre/yeyuxiang/projects/LightGCN-PyTorch/svd/svd_gcn/datasets/gowalla/svd_u.npy')).cuda()
        V = torch.Tensor(np.load('/mnt/lustre/yeyuxiang/projects/LightGCN-PyTorch/svd/svd_gcn/datasets/gowalla/svd_v.npy')).cuda()
        # print("svd split success", U.shape, V.shape, value.shape)

        self.user_vector = U[:, :self.req_vec] * self.weight_func(value[:self.req_vec])  # (user_num, 400)
        self.item_vector = V[:, :self.req_vec] * self.weight_func(value[:self.req_vec])  # (item_num, 400)
        self.latent_size = 64
        self.FS = nn.Parameter(
            torch.nn.init.uniform_(torch.randn(self.req_vec, self.latent_size),
                                   -np.sqrt(6. / (self.req_vec + self.latent_size)),
                                   np.sqrt(6. / (self.req_vec + self.latent_size))).cuda())

        # user-user, item-item graph
        self.user_matrix = ((self.rate_matrix.mm(self.rate_matrix.t())) != 0).float()
        self.item_matrix = ((self.rate_matrix.t().mm(self.rate_matrix)) != 0).float()


    def getUsersRating(self, users):
        # print("getUsersRating begin")
        final_user = self.user_vector.mm(self.FS)[users.long()]
        final_item = self.item_vector.mm(self.FS)
        rating = (final_user.mm(final_item.t())).sigmoid()# - self.rate_matrix * 1000
        return rating
    
    def getEmbedding(self, u, p, nega):
        batch_size = len(u)
        u = np.random.randint(0, self.num_users, batch_size)
        p = torch.multinomial(self.rate_matrix[u], 1, True).squeeze(1)
        nega = torch.multinomial(1 - self.rate_matrix[u], 1, True).squeeze(1)
        up = torch.multinomial(self.user_matrix[u], 1, True).squeeze(1)
        un = torch.multinomial(1 - self.user_matrix[u], 1, True).squeeze(1)
        pp = torch.multinomial(self.item_matrix[p], 1, True).squeeze(1)
        pn = torch.multinomial(1 - self.item_matrix[p], 1, True).squeeze(1)

        final_user, final_pos, final_nega = self.user_vector[u].mm(self.FS), \
                                            self.item_vector[p].mm( self.FS), \
                                            self.item_vector[nega].mm(self.FS)

        final_user_p, final_user_n = self.user_vector[up].mm(self.FS), self.user_vector[un].mm(self.FS)
        final_pos_p, final_pos_n = self.item_vector[pp].mm(self.FS), self.item_vector[pn].mm(self.FS)

        return final_user,final_pos,final_nega, final_user_p, final_user_n, final_pos_p, final_pos_n
    
    def bpr_loss(self, users, pos, neg):
        bs = len(users)
        (final_user,final_pos,final_nega, final_user_p, final_user_n, final_pos_p, final_pos_n) = self.getEmbedding(users.long(), pos.long(), neg.long())
        out = ((final_user * final_pos).sum(1) - (final_user * final_nega).sum(1)).sigmoid()
        self_loss_u = torch.log(((final_user * final_user_p).sum(1) - (final_user * final_user_n).sum(1)).sigmoid()).sum()
        self_loss_i = torch.log(((final_pos * final_pos_p).sum(1) - (final_pos * final_pos_n).sum(1)).sigmoid()).sum()
        regu_term =  (
                    final_user ** 2 +
                    final_pos ** 2 +
                    final_nega ** 2 +
                    final_user_p ** 2 +
                    final_user_n ** 2 +
                    final_pos_p ** 2 +
                    final_pos_n ** 2).sum()

        reg_loss = regu_term / bs
        loss = (-torch.log(out).sum()-self.coef_u*self_loss_u-self.coef_i*self_loss_i) / bs
        # print("loss: ", loss, reg_loss)
        # loss_all = loss + 0.01 * reg_loss
        # loss_all.backward()
        # with torch.no_grad():
        #     self.FS -= 9 * self.FS.grad
        #     self.FS.grad.zero_()
        return loss, reg_loss

