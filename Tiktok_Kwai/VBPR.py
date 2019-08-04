import torch
import torch.nn as nn
from torch.nn import init
import random
import numpy as np
import math
from tqdm import tqdm
from torch_geometric.utils import scatter_

class VBPR_model(nn.Module):
    def __init__(self, dim_v, v_feat, words_tensor, num_user, num_video, latent_dim):
        super(VBPR_model, self).__init__()
        self.u_embed_layer = nn.Embedding(num_user, latent_dim)
        nn.init.xavier_normal_(self.u_embed_layer.weight)
        self.v_embed_layer = nn.Embedding(num_video, latent_dim)
        nn.init.xavier_normal_(self.v_embed_layer.weight)
        self.u_feat_layer = nn.Embedding(num_user, latent_dim)
        nn.init.xavier_normal_(self.u_feat_layer.weight)
        self.v_feat_layer = nn.Embedding(num_video, dim_v)
        self.v_feat_layer.weight = nn.Parameter(torch.tensor(v_feat, dtype=torch.float, requires_grad=False))
        self.word_embedding = nn.Embedding(torch.max(words_tensor[1])+1, 128)
        nn.init.xavier_normal_(self.u_feat_layer.weight)
        self.tranfer_layer = nn.Linear(2*dim_v, latent_dim)
        self.v_base_layer = nn.Embedding(num_video, 1)
        nn.init.xavier_normal_(self.v_base_layer.weight)

        self.visual_bias_term = torch.zeros([latent_dim, 1], dtype=torch.float)

        self.num_user = num_user
        self.num_video = num_video
        self.latent_dim = latent_dim
        self.dim_v = dim_v
        self.words_tensor = words_tensor.cuda()

    def forward(self, batch_user, batch_video):
        batch_video -= self.num_user
        batch_u_embed = self.u_embed_layer(batch_user)
        batch_u_feat = self.u_feat_layer(batch_user)
        batch_v_embed = self.v_embed_layer(batch_video)

        self.t_feat = torch.tensor(scatter_('mean', self.word_embedding(self.words_tensor[1]), self.words_tensor[0])).cuda()

        batch_v_feat = nn.functional.relu(self.tranfer_layer(torch.cat((self.v_feat_layer(batch_video), self.t_feat[batch_video]), dim=1)))        
        score_embed = torch.sum(batch_u_embed*batch_v_embed, dim=1).view(-1,1)
        score_feat = torch.sum(batch_u_feat*batch_v_feat, dim=1).view(-1,1)
        score_base = self.v_base_layer(batch_video)
        score_visual_bias = torch.sum(torch.matmul(batch_v_feat, self.visual_bias_term.cuda()))

        return score_embed + score_feat + score_base + score_visual_bias


    def loss(self, data):
        user, pos_items, neg_items = data
        pos_scores = self.forward(user.cuda(), pos_items.cuda())
        neg_scores = self.forward(user.cuda(), neg_items.cuda()) 
        loss_value = -torch.sum(torch.log2(torch.sigmoid(pos_scores - neg_scores)))
        return loss_value


    def accuracy(self, dataset, neg_num=1000, topk=10):
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

            pos_len = len(pos_items)
            batch_user_tensor = torch.tensor(user).cuda()
            re_batch_user_pos = batch_user_tensor.repeat(batch_pos_tensor.size(0))
            re_batch_user_neg = batch_user_tensor.repeat(batch_neg_tensor.size(0))

            pos_score = self.forward(re_batch_user_pos, batch_pos_tensor)
            neg_score = self.forward(re_batch_user_neg, batch_neg_tensor)

            _, index_of_rank_list = torch.topk(torch.cat((neg_score, pos_score), dim=0).squeeze(), topk)
            index_set = set([iofr.cpu().item() for iofr in index_of_rank_list])

            num_hit = len(index_set.difference(all_set))

            sum_pre += float(num_hit/topk)
            sum_recall += float(num_hit/pos_len)

            ndcg_score = 0.0
            for i in range(pos_len):
                label_pos = neg_num + i
                if label_pos in index_of_rank_list:
                    index = list(index_of_rank_list.cpu().numpy()).index(label_pos)
                    ndcg_score = ndcg_score + math.log(2) / math.log(index + 2)
            sum_ndcg += ndcg_score/pos_len
        bar.close()
        return sum_pre/sum_item, sum_recall/sum_item, sum_ndcg/sum_item
