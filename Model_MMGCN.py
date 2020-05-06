import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import Parameter
from BaseModel import BaseModel
from torch_geometric.utils import scatter_

class GCN(torch.nn.Module):
    def __init__(self, edge_index, batch_size, num_user, num_item, dim_feat, dim_id, aggr_mode, concate, num_layer, has_id, dim_latent=None):
        super(GCN, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.dim_id = dim_id
        self.dim_feat = dim_feat
        self.dim_latent = dim_latent
        self.edge_index = edge_index
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

    def forward(self, features, id_embedding):
        temp_features = self.MLP(features) if self.dim_latent else features

        x = torch.cat((self.preference, temp_features),dim=0)
        x = F.normalize(x).cuda()

        h = F.leaky_relu(self.conv_embed_1(x, self.edge_index))#equation 1
        x_hat = F.leaky_relu(self.linear_layer1(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer1(x))#equation 5 
        x = F.leaky_relu(self.g_layer1(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer1(h)+x_hat)

        h = F.leaky_relu(self.conv_embed_2(x, self.edge_index))#equation 1
        x_hat = F.leaky_relu(self.linear_layer2(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer2(x))#equation 5
        x = F.leaky_relu(self.g_layer2(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer2(h)+x_hat)

        h = F.leaky_relu(self.conv_embed_3(x, self.edge_index))#equation 1
        x_hat = F.leaky_relu(self.linear_layer3(x)) + id_embedding if self.has_id else F.leaky_relu(self.linear_layer3(x))#equation 5
        x = F.leaky_relu(self.g_layer3(torch.cat((h, x_hat), dim=1))) if self.concate else F.leaky_relu(self.g_layer3(h)+x_hat)

        return x


class Net(torch.nn.Module):
    def __init__(self, v_feat, a_feat, t_feat, words_tensor, edge_index, batch_size, num_user, num_item, aggr_mode, concate, num_layer, has_id, user_item_dict, reg_weight, dim_x):
        super(Net, self).__init__()
        self.batch_size = batch_size
        self.num_user = num_user
        self.num_item = num_item
        self.aggr_mode = aggr_mode
        self.concate = concate
        self.user_item_dict = user_item_dict
        self.weight = torch.tensor([[1.0],[-1.0]]).cuda()
        self.reg_weight = reg_weight
        
        self.edge_index = torch.tensor(edge_index).t().contiguous().cuda()
        self.edge_index = torch.cat((self.edge_index, self.edge_index[[1,0]]), dim=1)
        self.num_modal = 0

        self.v_feat = torch.tensor(v_feat,dtype=torch.float).cuda()
        self.v_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.v_feat.size(1), dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id, dim_latent=256)

        self.a_feat = torch.tensor(a_feat,dtype=torch.float).cuda()
        self.a_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.a_feat.size(1), dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id)

        self.t_feat = torch.tensor(t_feat,dtype=torch.float).cuda()
        self.t_gcn = GCN(self.edge_index, batch_size, num_user, num_item, self.t_feat.size(1), dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id)

        # self.words_tensor = torch.tensor(words_tensor, dtype=torch.long).cuda()
        # self.word_embedding = nn.Embedding(torch.max(self.words_tensor[1])+1, 128)
        # nn.init.xavier_normal_(self.word_embedding.weight) 
        # self.t_gcn = GCN(self.edge_index, batch_size, num_user, num_item, 128, dim_x, self.aggr_mode, self.concate, num_layer=num_layer, has_id=has_id)

        self.id_embedding = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x), requires_grad=True)).cuda()
        self.result = nn.init.xavier_normal_(torch.rand((num_user+num_item, dim_x))).cuda()


    def forward(self):
        v_rep = self.v_gcn(self.v_feat, self.id_embedding)
        a_rep = self.a_gcn(self.a_feat, self.id_embedding)

        # # self.t_feat = torch.tensor(scatter_('mean', self.word_embedding(self.words_tensor[1]), self.words_tensor[0])).cuda()
        t_rep = self.t_gcn(self.t_feat, self.id_embedding)
        
        representation = (v_rep+a_rep+t_rep)/3

        self.result = representation
        return representation

    def loss(self, user_tensor, item_tensor):
        user_tensor = user_tensor.view(-1)
        item_tensor = item_tensor.view(-1)
        out = self.forward()
        user_score = out[user_tensor]
        item_score = out[item_tensor]
        score = torch.sum(user_score*item_score, dim=1).view(-1, 2)
        loss = -torch.mean(torch.log(torch.sigmoid(torch.matmul(score, self.weight))))
        reg_embedding_loss = (self.id_embedding[user_tensor]**2 + self.id_embedding[item_tensor]**2).mean()+(self.v_gcn.preference**2).mean()
        reg_loss = self.reg_weight * (reg_embedding_loss)
        return loss+reg_loss, reg_loss, loss, reg_embedding_loss, reg_embedding_loss

    def accuracy(self, step=2000, topk=10):
        user_tensor = self.result[:self.num_user]
        item_tensor = self.result[self.num_user:]

        start_index = 0
        end_index = self.num_user if step==None else step

        all_index_of_rank_list = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

            _, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()+self.num_user), dim=0)
            start_index = end_index
            
            if end_index+step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user

        length = self.num_user      
        precision = recall = ndcg = 0.0

        for row, col in self.user_item_dict.items():
            user = row
            pos_items = set(col)
            num_pos = len(pos_items)
            items_list = all_index_of_rank_list[user].tolist()

            items = set(items_list)

            num_hit = len(pos_items.intersection(items))
            
            precision += float(num_hit / topk)
            recall += float(num_hit / num_pos)

            ndcg_score = 0.0
            max_ndcg_score = 0.0

            for i in range(min(num_hit, topk)):
                max_ndcg_score += 1 / math.log2(i+2)
            if max_ndcg_score == 0:
                continue
                
            for i, temp_item in enumerate(items_list):
                if temp_item in pos_items:
                    ndcg_score += 1 / math.log2(i+2)

            ndcg += ndcg_score/max_ndcg_score

        return precision/length, recall/length, ndcg/length



    def full_accuracy(self, val_data, step=2000, topk=10):
        user_tensor = self.result[:self.num_user]
        item_tensor = self.result[self.num_user:]

        start_index = 0
        end_index = self.num_user if step==None else step

        all_index_of_rank_list = torch.LongTensor([])
        while end_index <= self.num_user and start_index < end_index:
            temp_user_tensor = user_tensor[start_index:end_index]
            score_matrix = torch.matmul(temp_user_tensor, item_tensor.t())

            for row, col in self.user_item_dict.items():
                if row >= start_index and row < end_index:
                    row -= start_index
                    col = torch.LongTensor(list(col))-self.num_user
                    score_matrix[row][col] = 1e-5

            _, index_of_rank_list = torch.topk(score_matrix, topk)
            all_index_of_rank_list = torch.cat((all_index_of_rank_list, index_of_rank_list.cpu()+self.num_user), dim=0)
            start_index = end_index
            
            if end_index+step < self.num_user:
                end_index += step
            else:
                end_index = self.num_user

        length = 0        
        precision = recall = ndcg = 0.0

        for data in val_data:
            user = data[0]
            pos_items = set(data[1:])
            num_pos = len(pos_items)
            if num_pos == 0:
                continue
            length += 1
            items_list = all_index_of_rank_list[user].tolist()

            items = set(items_list)

            num_hit = len(pos_items.intersection(items))
            
            precision += float(num_hit / topk)
            recall += float(num_hit / num_pos)

            ndcg_score = 0.0
            max_ndcg_score = 0.0

            for i in range(min(num_pos, topk)):
                max_ndcg_score += 1 / math.log2(i+2)
            if max_ndcg_score == 0:
                continue
                
            for i, temp_item in enumerate(items_list):
                if temp_item in pos_items:
                    ndcg_score += 1 / math.log2(i+2)

            ndcg += ndcg_score/max_ndcg_score

        return precision/length, recall/length, ndcg/length
