import random
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader

class MyDataset(Dataset):
	def __init__(self, path, num_user, num_item):
		super(MyDataset, self).__init__()
		self.data = np.load(path+'train.npy')
		self.adj_lists = np.load(path+'final_adj_dict.npy').item()
		self.all_set = set(range(num_user, num_user+num_item))

	def __getitem__(self, index):
		user, pos_item = self.data[index]
		neg_item = random.sample(self.all_set.difference(self.adj_lists[user]), 1)[0]
		return [user, pos_item, neg_item]

	def __len__(self):
		return len(self.data)



if __name__ == '__main__':
	num_item = 5986
	num_user = 55485
	dataset = MyDataset('./Data/', num_user, num_item)
	dataloader = DataLoader(dataset, batch_size=128, shuffle=True)

	for data in dataloader:
		user, pos_items, neg_items = data
		print(user)


