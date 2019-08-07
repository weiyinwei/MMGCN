import math
from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from BaseModel import BaseModel


class GCN(torch.nn.Module):
    def __init__(self, features, edge_index, batch_size, num_user, num_item, dim_id, aggr_mode, concate, num_layer, has_id, dim_latent=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = features.size(1)
        self.dim_latent = dim_latent
        self.edge_index = edge_index
        self.features = features
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.num_layer = num_layer
        self.has_id = has_id

        if self.dim_latent:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_latent), requires_grad=True)).cuda()
            self.MLP = nn.Linear(self.dim_feat, self.dim_latent)
            self.conv_embed_1 = BaseModel(self.dim_latent, self.dim_latent, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_latent, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_latent+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_latent, self.dim_id)    
            nn.init.xavier_normal_(self.g_layer1.weight) 

        else:
            self.preference = nn.init.xavier_normal_(torch.rand((num_user, self.dim_feat), requires_grad=True)).cuda()
            self.conv_embed_1 = BaseModel(self.dim_feat, self.dim_feat, aggr=self.aggr_mode)
            nn.init.xavier_normal_(self.conv_embed_1.weight)
            self.linear_layer1 = nn.Linear(self.dim_feat, self.dim_id)
            nn.init.xavier_normal_(self.linear_layer1.weight)
            self.g_layer1 = nn.Linear(self.dim_feat+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_feat, self.dim_id)     
            nn.init.xavier_normal_(self.g_layer1.weight)              
          
        self.conv_embed_2 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_2.weight)
        self.linear_layer2 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer2.weight)
        self.g_layer2 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)    

        self.conv_embed_3 = BaseModel(self.dim_id, self.dim_id, aggr=self.aggr_mode)
        nn.init.xavier_normal_(self.conv_embed_3.weight)
        self.linear_layer3 = nn.Linear(self.dim_id, self.dim_id)
        nn.init.xavier_normal_(self.linear_layer3.weight)
        self.g_layer3 = nn.Linear(self.dim_id+self.dim_id, self.dim_id) if self.concate else nn.Linear(self.dim_id, self.dim_id)    

    def forward(self, id_embedding):
        temp_features = self.MLP(self.features) if self.dim_latent else self.features

        x = torch.cat((self.preference, temp_features),dim=0)
        x = F.normalize(x).cuda()

        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))#equation 1
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer1(x))#equation 5 
        x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer1(h)+x_hat)

        if self.num_layer > 1:
            h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))#equation 1
            x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer2(x))#equation 5
            x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer2(h)+x_hat)
        if self.num_layer > 2:
            h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))#equation 1
            x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer3(x))#equation 5
            x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer3(h)+x_hat)
        return x


class MMGCN(torch.nn.Module):
    def __init__(self, features, edge_index, batch_size, num_user, num_item, aggr_mode, concate, num_layer, has_id, dim_x):
        super(MMGCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.concate = concate

        self.edge_index = torch.tensor(edge_index).t().contiguous().cuda()
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)
        
        v_feat, a_feat, t_feat = features
        self.v_feat = torch.tensor(v_feat, dtype=torch.float).cuda()
        self.a_feat = torch.tensor(a_feat, dtype=torch.float).cuda()
        self.t_feat = torch.tensor(t_feat, dtype=torch.float).cuda()

        self.v_gcn = GCN(self.v_feat, self.edge_index, batch_size, num_user, num_item, dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, dim_latent=128)#256)
        self.a_gcn = GCN(self.a_feat, self.edge_index, batch_size, num_user, num_item, dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, dim_latent=128)
        self.t_gcn = GCN(self.t_feat, self.edge_index, batch_size, num_user, num_item, dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, dim_latent=100)

        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x), requires_grad=True)).cuda()
        self.result_embed = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x))).cuda()


    def forward(self, user_nodes, pos_item_nodes, neg_item_nodes):
        v_rep = self.v_gcn(self.id_embedding)
        a_rep = self.a_gcn(self.id_embedding)
        t_rep = self.t_gcn(self.id_embedding)
        representation = (v_rep+a_rep+t_rep)/3
        self.result_embed = representation
        user_tensor = representation[user_nodes]
        pos_item_tensor = representation[pos_item_nodes]
        neg_item_tensor = representation[neg_item_nodes]
        pos_scores = torch.sum(user_tensor*pos_item_tensor, dim=1)
        neg_scores = torch.sum(user_tensor*neg_item_tensor, dim=1)
        return pos_scores, neg_scores


    def loss(self, data):
        user, pos_items, neg_items = data
        pos_scores, neg_scores = self.forward(user.cuda(), pos_items.cuda(), neg_items.cuda())
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

