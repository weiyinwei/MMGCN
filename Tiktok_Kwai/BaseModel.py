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
		if bias:
			self.bias = Parameter(torch.Tensor(out_channels))
		else:
			self.register_parameter('bias', None)
		self.reset_parameters()

	def reset_parameters(self):
		uniform(self.in_channels, self.weight)
		uniform(self.in_channels, self.bias)

	def forward(self, x, edge_index, size=None):
		if size is None:
			edge_index, _ = remove_self_loops(edge_index)
			edge_index, _ = add_self_loops(edge_index, num_nodes=x.size(0))
		x = x.unsqueeze(-1) if x.dim() == 1 else x
		x = torch.matmul(x, self.weight)
		return self.propagate(edge_index, size=(x.size(0), x.size(0)), x=x)

	def message(self, x_j, edge_index, size):
		# if self.aggr == 'add':
		# 	row, col = edge_index
		# 	deg = degree(row, size[0], dtype=x_j.dtype)
		# 	deg_inv_sqrt = deg.pow(-0.5)
		# 	norm = deg_inv_sqrt[row] * deg_inv_sqrt[col]
		# 	return norm.view(-1, 1) * x_j
		return x_j

	def update(self, aggr_out):
		if self.bias is not None:
			aggr_out = aggr_out + self.bias
		if self.normalize:
			aggr_out = F.normalize(aggr_out, p=2, dim=-1)
		return aggr_out

	def __repr(self):
		return '{}({},{})'.format(self.__class__.__name__, self.in_channels, self.out_channels)
