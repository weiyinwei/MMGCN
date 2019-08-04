import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data
from BaseModel import BaseModel

class MMGraphSAGE(torch.nn.Module):
    def __init__(self, features, edge_index, batch_size, num_user, num_item, dim_latent, dim_feat):
        super(MMGraphSAGE, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item

        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = torch.tensor(edge_index).t().contiguous().cuda()
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)
        self.features = features.cuda()
        self.features.requires_grad = False
        self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).cuda()
        self.result_embed = nn.init.xavier_normal_(torch.rand((num_user+num_item, self.dim_latent))).cuda()

        self.linear_layer1 = nn.Linear(self.dim_feat, 1024)
        nn.init.xavier_normal_(self.linear_layer1.weight)
        self.linear_layer2 = nn.Linear(1024, self.dim_latent)
        nn.init.xavier_normal_(self.linear_layer2.weight)

        self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent)
        nn.init.xavier_normal_(self.conv_embed_1.weight)        
        self.conv_embed_2 = BaseModel(self.dim_latent, self.dim_latent)
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        # self.conv_embed_3 = BaseModel(self.dim_latent, self.dim_latent)
        # nn.init.xavier_normal_(self.conv_embed_3.weight)

    def forward(self, user_nodes, item_nodes):
        temp_features = F.leaky_relu(self.linear_layer1(self.features))
        temp_features = F.leaky_relu(self.linear_layer2(temp_features))
        x = torch.cat((self.preference, temp_features),dim=0)
        x = F.normalize(x).cuda()
        x = F.leaky_relu(self.conv_embed_1(x, self.edge_index))
        x = F.leaky_relu(self.conv_embed_2(x, self.edge_index))
        # x = F.leaky_relu(self.conv_embed_3(x, self.edge_index))

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