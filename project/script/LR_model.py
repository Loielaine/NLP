from torch import nn
import torch.nn.functional as F

class LogisticRegression(nn.Module):
	def __init__(self, input_size, num_classes):
		super(LogisticRegression, self).__init__()
		self.linear = nn.Linear(input_size, 200)
		self.output = nn.Linear(200, num_classes)
			# Initialize the weights of both layers
		self.init_weights()

	def init_weights(self):
		initrange = 0.1
		self.linear.weight.data.uniform_(-1 * initrange, initrange)
		self.linear.bias.data.zero_()
		self.output.weight.data.uniform_(-1 * initrange, initrange)
		self.output.bias.data.zero_()

	def forward(self, x):
		x = x.view(x.size()[0], -1)
		input = self.linear(x)
		output = self.output(input)
		out = F.softmax(output)
		return out
