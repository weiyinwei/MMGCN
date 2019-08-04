import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from BaseModel import BaseModel
from torch_geometric.utils import scatter_

class GCMC(torch.nn.Module):
    def __init__(self, v_feat, words_tensor, edge_index, batch_size, num_user, num_item, dim_latent, dim_feat):
        super(GCMC, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item

        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = torch.tensor(edge_index).t().contiguous().cuda()
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)
        self.v_feat = torch.tensor(v_feat, requires_grad=False, dtype=torch.float).cuda()
        self.words_tensor = words_tensor.cuda()

        self.word_embedding = nn.Embedding(torch.max(self.words_tensor)+1, 128)
        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, self.dim_latent), requires_grad=True)).cuda()
        # nn.register_parameter('id_embedding', self.id_embedding)
        self.result_embed = nn.init.xavier_normal_(torch.rand((num_user+num_item, self.dim_latent))).cuda()

        self.linear_layer = nn.Linear(self.dim_feat+128, self.dim_latent)
        nn.init.xavier_normal_(self.linear_layer.weight)

        self.conv_embed = BaseModel(self.dim_latent, self.dim_latent).cuda()
        nn.init.xavier_normal_(self.conv_embed.weight)     

        self.weight_W = nn.init.xavier_normal_(torch.rand((self.dim_latent, self.dim_latent))).cuda()
        self.weight_2 = nn.init.xavier_normal_(torch.rand((self.dim_latent, self.dim_latent))).cuda()



    def forward(self, user_nodes, item_nodes):
        x = F.normalize(self.id_embedding).cuda()
        x = F.leaky_relu(self.conv_embed(x, self.edge_index))
        
        self.t_feat = torch.tensor(scatter_('mean', self.word_embedding(self.words_tensor[1]), self.words_tensor[0])).cuda()
        f = F.leaky_relu(self.linear_layer(torch.cat((self.v_feat, self.t_feat), dim=1)))
        f_hat = torch.matmul(f, self.weight_2)
        f_hat = torch.cat((torch.zeros(self.num_user, self.dim_latent).cuda(),f_hat), dim=0)
        x = F.leaky_relu(torch.matmul(x, self.weight_W)+f_hat)

        self.result_embed = x
        user_tensor = x[user_nodes]
        item_tensor = x[item_nodes]
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

            user_embed = self.result_embed[batch_user_tensor]
            pos_v_embed = self.result_embed[batch_pos_tensor]
            neg_v_embed = self.result_embed[batch_neg_tensor]

            num_pos = len(pos_items)
            pos_score = torch.sum(pos_v_embed*user_embed, dim=1)
            neg_score = torch.sum(neg_v_embed*user_embed, dim=1)

            _, index_of_rank_list = torch.topk(torch.cat((neg_score, pos_score)), topk)
            index_set = set([iofr.cpu().item() for iofr in index_of_rank_list])
            num_hit = len(index_set.difference(all_set))
            # print(num_hit)
            sum_pre += float(num_hit/topk)
            sum_recall += float(num_hit/num_pos)
            # print(num_pos)
            ndcg_score = 0.0
            for i in range(num_pos):
                label_pos = neg_num + i
                if label_pos in index_of_rank_list:
                    index = list(index_of_rank_list.cpu().numpy()).index(label_pos)
                    ndcg_score = ndcg_score + math.log(2) / math.log(index + 2)
            sum_ndcg += ndcg_score/num_pos
            # print(sum_ndcg, sum_item)
        bar.close()

        return sum_pre/sum_item, sum_recall/sum_item, sum_ndcg/sum_item