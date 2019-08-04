import os
import argparse
import time
import numpy as np
from tqdm import tqdm
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from DataLoad import MyDataset
from Model import MMGCN
from VBPR import VBPR_model
from MMGraphSAGE import MMGraphSAGE
from NGCF import NGCF

class Net:
    def __init__(self, args):
        ##########################################################################################################################################
        seed = args.seed
        np.random.seed(seed)
        random.seed(seed)
        self.device = torch.device("cuda:0" if torch.cuda.is_available() and not args.no_cuda else "cpu")
        ##########################################################################################################################################
        self.model_name = args.model_name
        self.data_path = args.data_path
        self.learning_rate = args.l_r#l_r#
        self.weight_decay = args.weight_decay#weight_decay#
        self.batch_size = args.batch_size
        self.concat = args.concat
        self.num_workers = args.num_workers
        self.num_epoch = args.num_epoch
        self.num_user = args.num_user
        self.num_item = args.num_item
        self.dim_latent = args.dim_latent
        self.aggr_mode = args.aggr_mode#aggr_mode#
        self.num_layer = args.num_layer
        self.has_id = args.has_id
        self.dim_v = 128
        self.dim_a = 128
        self.dim_t = 100
        ##########################################################################################################################################
        print('Data loading ...')
        self.train_dataset = MyDataset('./Data/', self.num_user, self.num_item)
        self.train_dataloader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        self.edge_index = np.load('./Data/train.npy')
        self.val_dataset = np.load('./Data/val.npy')
        self.test_dataset = np.load('./Data/test.npy')
        self.v_feat_tensor = torch.load('./Data/visual_feat.pt')
        self.a_feat_tensor = torch.load('./Data/audio_feat.pt')
        self.t_feat_tensor = torch.load('./Data/textual_feat.pt')
        print('Data has been loaded.')
        ##########################################################################################################################################
        if self.model_name == 'MMGCN':
            self.model = MMGCN(self.v_feat_tensor, self.a_feat_tensor, self.t_feat_tensor, self.edge_index, self.batch_size, self.num_user, self.num_item, self.aggr_mode, self.concat, self.num_layer, self.has_id, self.dim_latent).cuda()
        
        elif self.model_name == 'VBPR':
            self.model = VBPR_model(self.v_feat_tensor, self.a_feat_tensor, self.t_feat_tensor, self.num_user, self.num_item, self.dim_latent).to(self.device)

        elif self.model_name == 'GraphSAGE':
            self.model = MMGraphSAGE(self.v_feat_tensor, self.a_feat_tensor, self.t_feat_tensor, self.edge_index, self.batch_size, self.num_user, self.num_item, self.dim_latent, self.dim_feat).cuda()

        elif self.model_name == 'NGCF':
            self.model = NGCF(self.v_feat_tensor, self.a_feat_tensor, self.t_feat_tensor, self.edge_index, self.batch_size, self.num_user, self.num_item, self.dim_latent).cuda()

        if args.PATH_weight_load and os.path.exists(args.PATH_weight_load):
            self.model.load_state_dict(torch.load(args.PATH_weight_load))
            print('module weights loaded....')
        ##########################################################################################################################################
        self.optimizer = torch.optim.Adam([{'params': self.model.parameters(), 'lr': self.learning_rate}], weight_decay=self.weight_decay)
        ##########################################################################################################################################

    def run(self):
        max_recall = 0.0
        max_pre = 0.0
        max_ndcg = 0.0
        step = 0
        for epoch in range(self.num_epoch):
            self.model.train()
            print('Now, training start ...')
            pbar = tqdm(total=len(self.train_dataset))
            sum_loss = 0.0
            for data in self.train_dataloader:
                self.optimizer.zero_grad()
                self.loss = self.model.loss(data)
                self.loss.backward()
                self.optimizer.step()
                pbar.update(self.batch_size)
                sum_loss += self.loss
            print(sum_loss/self.batch_size)
            pbar.close()

            if epoch % 5 == 0:
                print('Validation start...')
                self.model.eval()
                with torch.no_grad():
                    precision, recall, ndcg_score = self.model.accuracy(self.val_dataset, topk=10)
                    print('---------------------------------Val: {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                        epoch, 10, precision, recall, ndcg_score))
            if epoch % 10 == 0:
                self.model.eval()
                with torch.no_grad():
                    precision, recall, ndcg_score = self.model.accuracy(self.test_dataset, topk=10)
                    print('---------------------------------Test: {0}-th epoch {1}-th top Precition:{2:.4f} Recall:{3:.4f} NDCG:{4:.4f}---------------------------------'.format(
                        epoch, 10, precision, recall, ndcg_score))

        return max_recall, max_pre, max_ndcg


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--no-cuda', action='store_true', default=False, help='Disables CUDA training.')
    parser.add_argument('--model_name', default='MMGCN', help='Model name.')
    parser.add_argument('--data_path', default='amazon-book', help='Dataset path')
    parser.add_argument('--PATH_weight_load', default=None, help='Loading weight filename.')
    parser.add_argument('--PATH_weight_save', default=None, help='Writing weight filename.')
    parser.add_argument('--l_r', type=float, default=1e-3, help='Learning rate.')
    parser.add_argument('--weight_decay', type=float, default=1e-3, help='Weight decay.')
    parser.add_argument('--batch_size', type=int, default=1024, help='Batch size.')
    parser.add_argument('--dim_latent', type=int, default=64, help='Latent dimension.')
    parser.add_argument('--num_epoch', type=int, default=200, help='Epoch number.')
    parser.add_argument('--num_workers', type=int, default=40, help='Workers number.')
    parser.add_argument('--num_user', type=int, default=36656, help='User number.')
    parser.add_argument('--num_item', type=int, default=76085, help='Item number.')
    parser.add_argument('--aggr_mode', default='mean', help='Aggregation mode.')
    parser.add_argument('--num_layer', type=int, default=2, help='Layer number.')
    parser.add_argument('--has_id', type=bool, default=True, help='Has id_embedding')
    parser.add_argument('--concat', type=bool, default=False, help='Concatenation')
    args = parser.parse_args()
    egcn = Net(args)
    egcn.run()    


