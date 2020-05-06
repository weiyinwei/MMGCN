import torch
import torch.nn.functional as F
from torch.nn import Parameter
from torch_geometric.nn.conv import MessagePassing
from torch_geometric.utils import remove_self_loops, add_self_loops, degree
from torch_geometric.nn.inits import uniform

class BaseModel(MessagePassing):
	def __init__(self, in_channels, out_channels, normalize=True, bias=True, aggr='add', **kwargs):
		super(BaseModel, self).__init__(aggr=aggr, **kwargs)
		self.aggr = aggr
		self.in_channels = in_channels
		self.out_channels = out_channels
		self.normalize = normalize
		self.weight = Parameter(torch.Tensor(self.in_channels, out_channels))

		self.reset_parameters()

	def reset_parameters(self):
		uniform(self.in_channels, self.weight)

	def forward(self, x, edge_index, size=None):
		x = torch.matmul(x, self.weight)
		return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

	def message(self, x_j, edge_index, size):
		return x_j

	def update(self, aggr_out):
		return aggr_out

	def __repr(self):
		return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
