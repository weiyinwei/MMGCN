import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree, scatter_
from torch_geometric.nn.inits import uniform

class NGCFConv(MessagePassing):    
    def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
        super(NGCFConv, self).__init__(aggr=aggr, **kwargs)
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.normalize = normalize
        self.weight1 = Parameter(torch.Tensor(self.in_channels, out_channels))
        self.weight2 = Parameter(torch.Tensor(self.in_channels, out_channels))
        if bias:
            self.bias = Parameter(torch.Tensor(out_channels))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        uniform(self.in_channels, self.weight1)
        uniform(self.in_channels, self.weight2)
        uniform(self.in_channels, self.bias)

    def forward(self, x, edge_index, size=None):
        edge_index, _ = remove_self_loops(edge_index)
        x = x.unsqueeze(-1) if x.dim() == 1 else x      
        self.x = x
        return self.propagate(edge_index, size=size, x=x)

    def message(self, x_i, x_j, edge_index, size):
        x_j = torch.matmul(x_j, self.weight1) + torch.matmul(x_i.mul_(x_j), self.weight2)
        row, col = edge_index
        deg = degree(row, size[0], dtype=x_j.dtype)
        deg_inv_sqrt = deg.pow(-0.5)
        norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
        return norm.view(-1, 1) * x_j
    
    def update(self, aggr_out):
        temp_x = torch.matmul(self.x, self.weight1)
        aggr_out += temp_x
        if self.bias is not None:
            aggr_out = aggr_out + self.bias        
        if self.normalize:
            aggr_out = F.normalize(aggr_out, p=2, dim=-1)
        return aggr_out
        
    def __repr(self):
        return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)


class NGCF(torch.nn.Module):
    def __init__(self, v_feat, a_feat, words_tensor, edge_index, batch_size, num_user, num_item, dim_latent, dim_word=128, dim_hidden=128):
        super(NGCF, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_hidden = dim_hidden
        self.dim_latent = dim_latent
        self.dim_feat = v_feat.size(1) + a_feat.size(1) + dim_word
        self.edge_index = torch.tensor(edge_index).t().contiguous().cuda()
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)
        self.v_feat = torch.tensor(v_feat, requires_grad=False, dtype=torch.float).cuda()
        self.a_feat = torch.tensor(a_feat, requires_grad=False, dtype=torch.float).cuda()
        self.words_tensor = words_tensor.cuda()
        self.word_embedding = nn.Embedding(torch.max(self.words_tensor)+1, dim_word)      
        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, self.dim_latent), requires_grad=True)).cuda()
        self.result_embed = nn.init.xavier_normal_(torch.rand((num_user+num_item, 3*self.dim_latent))).cuda()
        self.linear_layer1 = nn.Linear(self.dim_feat, 3*self.dim_latent)
        nn.init.xavier_normal_(self.linear_layer1.weight)
        self.conv_embed_1 = NGCFConv(self.dim_latent, self.dim_latent)
        self.conv_embed_2 = NGCFConv(self.dim_latent, self.dim_latent)
        self.conv_embed_3 = NGCFConv(self.dim_latent, self.dim_latent)

    def forward(self, user_nodes, item_nodes):
        x = F.normalize(self.id_embedding).cuda()
        x = F.leaky_relu_(self.conv_embed_1(x, self.edge_index))
        x1 = F.leaky_relu_(self.conv_embed_2(x, self.edge_index))
        x2 = F.leaky_relu(self.conv_embed_3(x1, self.edge_index))
        x = torch.cat((x, x1, x2), dim=1)
        self.result_embed = x
        user_tensor = x[user_nodes]
        self.t_feat = torch.tensor(scatter_('mean', self.word_embedding(self.words_tensor[1]), self.words_tensor[0])).cuda()
        item_feat = torch.cat((self.v_feat[item_nodes-self.num_user], self.a_feat[item_nodes-self.num_user], self.t_feat[item_nodes-self.num_user]), dim=1)
        item_tensor = x[item_nodes] + F.leaky_relu_(self.linear_layer1(item_feat))
        scores = torch.sum(user_tensor*item_tensor, dim=1)
        return scores

    def loss(self, data):
        user, pos_items, neg_items = data
        pos_scores = self.forward(user.cuda(), pos_items.cuda())
        neg_scores = self.forward(user.cuda(), neg_items.cuda()) 
        loss_value = -torch.sum(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        return loss_value

    def accuracy(self, dataset, topk=10, neg_num=1000):
        all_set = set(list(np.arange(neg_num)))
        sum_pre = 0.0
        sum_recall = 0.0
        sum_ndcg = 0.0
        sum_item = 0
        bar = tqdm(total=len(dataset))

        for data in dataset:
            bar.update(1)
            if len(data) < 1002:
                continue

            sum_item += 1
            user = data[0]
            neg_items = data[1:1001]
            pos_items = data[1001:]

            batch_user_tensor = torch.tensor(user).cuda() 
            batch_pos_tensor = torch.tensor(pos_items).cuda()
            batch_neg_tensor = torch.tensor(neg_items).cuda()

            pos_item_feat = torch.cat((self.v_feat[batch_pos_tensor-self.num_user], self.a_feat[batch_pos_tensor-self.num_user], self.t_feat[batch_pos_tensor-self.num_user]), dim=1)
            neg_item_feat = torch.cat((self.v_feat[batch_neg_tensor-self.num_user], self.a_feat[batch_neg_tensor-self.num_user], self.t_feat[batch_neg_tensor-self.num_user]), dim=1)
    
            user_embed = self.result_embed[batch_user_tensor]
            pos_v_embed = self.result_embed[batch_pos_tensor] + F.leaky_relu_(self.linear_layer1(pos_item_feat))
            neg_v_embed = self.result_embed[batch_neg_tensor] + F.leaky_relu_(self.linear_layer1(neg_item_feat))

            num_pos = len(pos_items)
            pos_score = torch.sum(pos_v_embed*user_embed, dim=1)
            neg_score = torch.sum(neg_v_embed*user_embed, dim=1)

            _, index_of_rank_list = torch.topk(torch.cat((neg_score, pos_score)), topk)
            index_set = set([iofr.cpu().item() for iofr in index_of_rank_list])
            num_hit = len(index_set.difference(all_set))
            sum_pre += float(num_hit/topk)
            sum_recall += float(num_hit/num_pos)
            ndcg_score = 0.0
            for i in range(num_pos):
                label_pos = neg_num + i
                if label_pos in index_of_rank_list:
                    index = list(index_of_rank_list.cpu().numpy()).index(label_pos)
                    ndcg_score = ndcg_score + math.log(2) / math.log(index + 2)
            sum_ndcg += ndcg_score/num_pos
        bar.close()

        return sum_pre/sum_item, sum_recall/sum_item, sum_ndcg/sum_item