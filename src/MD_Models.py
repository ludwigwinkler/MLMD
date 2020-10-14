import sys

import matplotlib

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch
from torch.nn import functional as F
from torch.nn import Sequential, Module
from torch.nn import Linear, LSTM, RNN
from torch.nn import Tanh, LeakyReLU, ReLU

from torchdiffeq import odeint_adjoint as odeint

Tensor = torch.Tensor

sys.path.append("../../..")

from DiffEqNets.DiffEqNets_ModelUtils import IntegratorWrapper
from DiffEqNets.DiffEqNets_ModelUtils import ODEWrapper, ODE2Wrapper, FirstOrderPDWrapper, SecOrderPDWrapper, HamiltonianWrapper
from DiffEqNets.DiffEqNets_ModelUtils import apply_rescaling, ForwardHook, ForwardPreHook

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class MD_Model:

	def __init__(self, hparams):

		pass

	def criterion(self, _pred, _target):

		assert _pred.shape==_target.shape
		assert torch.sum(_pred[:,0]- _target[:,0])==0, f" y0's aren't the same"

		return F.mse_loss(_pred, _target)#*_pred.shape[1]

class MD_BiModel:

	def __init__(self, hparams):
		# if hparams.lr <= 0:
		# 	self.optim = torch.optim.Adam(self.parameters())
		# elif hparams.lr > 0:
		# 	self.optim = torch.optim.Adam(self.parameters(), lr=hparams.lr)
		pass

	def criterion(self, _pred, _target):
		assert _pred.shape == _target.shape, f'{_pred.shape=} {_target.shape=}'
		assert torch.sum(_pred[:, 0] - _target[:, 0]) == 0, f" y[0] aren't the same"
		# assert torch.sum(_pred[:, -1] - _target[:, -1]) == 0, f" y[-1]'s aren't the same"

		return F.mse_loss(_pred, _target)  # *_pred.shape[1]

class MD_BiDirectional_RNN(Module, MD_BiModel):

	def __init__(self, hparams, scaling):

		Module.__init__(self)
		self.hparams = hparams

		self.input_embedding = Linear(self.hparams.in_features, self.hparams.num_hidden, bias=False)
		self.rnn = RNN(input_size=self.hparams.in_features, hidden_size=self.hparams.num_hidden, num_layers=self.hparams.num_layers, batch_first=True)
		self.out_emb = Linear(self.hparams.num_hidden, self.hparams.in_features, bias=False)

		self.y_mu 	= scaling['y_mu']
		self.y_std 	= scaling['y_std']
		self.dy_mu 	= scaling['dy_mu']
		self.dy_std 	= scaling['dy_std']

		MD_BiModel.__init__(self, hparams)

	def criterion(self, _pred, _target):

		# assert torch.sum(torch.abs(_pred[:, 0] - _target[:, 0])) == 0, f" y0's aren't the same"

		return F.mse_loss(_pred, _target)  # *_pred.shape[1]

	def forward(self, t, x):

		assert x.shape[1]==2*self.hparams.input_length
		assert x.dim()==3

		x1, x2 = torch.chunk(x, chunks=2, dim=1)

		x1_for = x1
		x2_back = x2.flip([1]) # flip the backward conditioning so that we can simply pass the data forward
		assert (x2_back[:,-1,:]==x2[:,0,:]).all()

		total_length = t + 2 * self.hparams.input_length
		lin_weights = (torch.arange(0, total_length, dtype=x.dtype, device=x.device)) / (total_length - 1)
		assert lin_weights[0] == 0 and lin_weights[-1] == 1, f'{lin_weights=}'

		weights = torch.stack([lin_weights.flip([0]), lin_weights], dim=-1)
		weights = weights / weights.sum(dim=-1, keepdim=True)
		weights = weights.unsqueeze(-2).to(x.device)

		'''
		Forwards
		'''
		pred_for = self.forward_t(t=t+x2.shape[1], x=x1_for)

		'''
		Backwards
		'''
		if True:
			velocity_flip = torch.ones(x2_back.shape[-1], device=x.device)
			velocity_flip[velocity_flip.shape[-1] // 2:] = -1
			x2_back = x2_back * velocity_flip
			pred_back = self.forward_t(t=t+x2_back.shape[1], x=x2_back).flip([1])
			pred_back = pred_back * velocity_flip
			assert F.mse_loss(pred_back[:, -1], x[:, -1]) == 0., f'{pred_back[0,-1]=} {x[0,-1]=}'

			pred = torch.stack([pred_for, pred_back], dim=-1)
			weights = weights.expand_as(pred)
			out = torch.sum(pred * weights, dim=-1)

			# out = torch.sum(pred*weights, dim=-1)
			# out = pred_back

		else:
			out = pred_for



		return out

	def forward_t(self, t, x):
		'''
		:param t:
		:param x: [x(0), x(1), x(2)]
		:return:
		'''
		# print(f'{x.dtype=} {self.rnn._parameters["weight_ih_l0"].dtype}')
		assert x.dim() == 3

		pred, h = self.rnn(x)
		'''
		[dx(0), dx(1), dx(2)] -> x(0) + dx(0) = x(1)
		'''
		dx = self.out_emb(pred)
		out = torch.cat([x[:,:1,:], x + dx], dim=1) # t = input_length + 1
		# print(f"{out.shape=}")
		# exit()
		# out = torch.cat([out, out[:,-1:,:] + dx[:,-1:,:]], dim=1)

		for step in range(t-1): # length of x already input_length+1
			pred_t, h = self.rnn(out[:,-1:], h)
			dx_t = self.out_emb(pred_t)
			# if self.dy_std is not None:
			# 	dx_t = self.dy_std * dx_t

			out = torch.cat([out, out[:, -1:, :] + dx_t], dim=1)
			# out = torch.cat([out, dx_t[:, -1:, :]], dim=1)

		return out

class MD_BiDirectional_LSTM(Module, MD_BiModel):

	def __init__(self, hparams, scaling):

		Module.__init__(self)
		self.hparams = hparams

		self.lstm = LSTM(input_size=self.hparams.in_features, hidden_size=hparams.num_hidden, num_layers=hparams.num_layers - 1,
				 batch_first=True)
		self.out_emb = Linear(self.hparams.num_hidden, self.hparams.in_features, bias=True)

		self.y_mu = scaling['y_mu']
		self.y_std = scaling['y_std']
		self.dy_mu = scaling['dy_mu']
		self.dy_std = scaling['dy_std']

		MD_BiModel.__init__(self, hparams)

	def criterion(self, _pred, _target):

		# assert torch.sum(torch.abs(_pred[:, 0] - _target[:, 0])) == 0, f" y0's aren't the same"

		return F.mse_loss(_pred, _target)  # *_pred.shape[1]

	def forward(self, t, x):

		assert x.shape[1]==2*self.hparams.input_length
		assert x.dim()==3

		x1, x2 = torch.chunk(x, chunks=2, dim=1)

		x1_for = x1
		x2_back = x2.flip([1]) # flip the backward conditioning so that we can simply pass the data forward
		assert (x2_back[:,-1,:]==x2[:,0,:]).all()

		total_length = t + 2 * self.hparams.input_length
		lin_weights = (torch.arange(0, total_length, dtype=x.dtype, device=x.device)) / (total_length - 1)
		assert lin_weights[0] == 0 and lin_weights[-1] == 1, f'{lin_weights=}'

		weights = torch.stack([lin_weights.flip([0]), lin_weights], dim=-1)
		weights = weights / weights.sum(dim=-1, keepdim=True)
		weights = weights.unsqueeze(-2).to(x.device)

		'''
		Forwards
		'''
		pred_for = self.forward_t(t=t+x2.shape[1], x=x1_for)

		'''
		Backwards
		'''
		if True:
			velocity_flip = torch.ones(x2_back.shape[-1], device=x.device)
			velocity_flip[velocity_flip.shape[-1] // 2:] = -1
			x2_back = x2_back * velocity_flip
			pred_back = self.forward_t(t=t+x2_back.shape[1], x=x2_back).flip([1])
			pred_back = pred_back * velocity_flip
			assert F.mse_loss(pred_back[:, -1], x[:, -1]) == 0., f'{pred_back[0,-1]=} {x[0,-1]=}'

			pred = torch.stack([pred_for, pred_back], dim=-1)
			weights = weights.expand_as(pred)
			out = torch.sum(pred * weights, dim=-1)

			# out = torch.sum(pred*weights, dim=-1)
			# out = pred_back

		else:
			out = pred_for



		return out

	def forward_t(self, t, x):
		'''
		:param t:
		:param x: [x(0), x(1), x(2)]
		:return:
		'''
		if x.dim() == 2: x = x.unsqueeze(1)
		assert x.dim() == 3

		pred, (h, c) = self.lstm(x)
		dx = self.out_emb(pred)
		out = torch.cat([x[:, :1, :], x + dx], dim=1)
		for step in range(t - 1):  # because we add the first entry y0 at the beginning
			pred_t, (h, c) = self.lstm(out[:, -1:], (h, c))
			dx_t = self.out_emb(pred_t)
			# if self.dy_std is not None:
			# 	dx_t = self.dy_std * dx_t

			out = torch.cat([out, out[:, -1:, :] + dx_t], dim=1)

		return out

class MD_BiDirectional_Hamiltonian(Module, MD_BiModel):

	def __init__(self, hparams, scaling):
		Module.__init__(self)

		self.hparams = hparams
		actfunc = torch.nn.Tanh
		bias = True
		net = Sequential(Linear(self.hparams.in_features, self.hparams.num_hidden, bias=True),
				 actfunc(),
				 Linear(self.hparams.num_hidden, self.hparams.num_hidden, bias=bias),
				 actfunc(),
				 Linear(self.hparams.num_hidden, self.hparams.num_hidden, bias=bias),
				 actfunc(),
				 Linear(self.hparams.num_hidden, self.hparams.num_hidden, bias=bias),
				 # actfunc(),
				 # Linear(hparams.num_hidden, hparams.num_hidden, bias=bias),
				 # actfunc(),
				 # Linear(hparams.num_hidden, hparams.num_hidden, bias=bias),
				 # actfunc(),
				 # Linear(hparams.num_hidden, hparams.num_hidden//2, bias=bias),
				 # actfunc(),
				 # Linear(hparams.num_hidden//2, hparams.num_hidden//4, bias=bias),
				 # actfunc(),
				 Linear(self.hparams.num_hidden, 1, bias=bias)
				 )

		# for module in net.modules():
		# 	module.apply(lambda x: apply_rescaling(x, scale=1.))

		self.net = HamiltonianWrapper(net=net)
		self.integrator = IntegratorWrapper(self.net, odeint_str=self.hparams.odeint)
		MD_BiModel.__init__(self, self.hparams)

	def forward(self, t, x):
		assert x.shape[1] == 2 * self.hparams.input_length, f'{x.shape=} {self.hparams.input_length=}'
		assert x.dim() == 3

		x1, x2 = torch.chunk(x, chunks=2, dim=1)
		assert x1.shape[1] == x2.shape[1] == 1
		x1 = x1.squeeze(1)
		x2 = x2.squeeze(1)

		assert x1.shape == x2.shape

		x1_for = x1
		x2_back = x2

		total_length = t + 2 * self.hparams.input_length
		lin_weights = (torch.arange(0, total_length, dtype=x.dtype, device=x.device)) / (total_length - 1)
		assert lin_weights[0] == 0 and lin_weights[-1] == 1, f'{lin_weights=}'

		weights = torch.stack([lin_weights.flip([0]), lin_weights], dim=-1)
		weights = weights / weights.sum(dim=-1, keepdim=True)
		weights = weights.unsqueeze(-2)

		'''
		Forwards
		'''
		pred_for = self.integrator(t=t + self.hparams.input_length, x=x1_for)
		assert F.mse_loss(pred_for[:, 0], x[:, 0]) == 0.

		'''
		Backwards
		[q, p] -> [q, -p]
		After integration flip from [x(-1), x(-2), x(-3)] to [x(-3), x(-2), x(-1)] for correct direction
		'''
		# print(x2_back.shape)

		velocity_flip = torch.ones(x2_back.shape[-1], device=x.device)
		velocity_flip[velocity_flip.shape[-1] // 2:] = -1
		x2_back = x2_back * velocity_flip

		pred_back = self.integrator(t=t + self.hparams.input_length, x=x2_back).flip(
			[-2])  # Solve [x_0, x_1, x_2 ... x_T] but flip it for [x_T, ..., x_2, x_1, x_0]
		pred_back = pred_back * velocity_flip  # flip velocity since it should face 'forward' in time

		assert F.mse_loss(pred_back[:, -1], x[:, -1]) == 0., f'{pred_back[0,-1]=} {x[0,-1]=}'

		pred = torch.stack([pred_for, pred_back], dim=-1)
		weights = weights.expand_as(pred)
		out = torch.sum(pred * weights, dim=-1)

		assert out.shape == (x.shape[0], total_length, x.shape[-1])

		return out

class MD_BiDirectional_ODE(Module, MD_BiModel):

	def __init__(self, hparams, scaling):
		Module.__init__(self)

		self.hparams = hparams

		self.y_mu = scaling['y_mu']
		self.y_std = scaling['y_std']
		self.dy_mu = scaling['dy_mu']
		self.dy_std = scaling['dy_std']

		bias = True
		net = Sequential()
		net.add_module(name='Layer_0', module=Linear(self.hparams.in_features, self.hparams.num_hidden, bias=bias))
		net.add_module(name='Activ_0', module=ReLU())

		for layer in range(self.hparams.num_layers):
			net.add_module(name=f'Layer_{layer + 1}', module=Linear(self.hparams.num_hidden, self.hparams.num_hidden, bias=bias))
			net.add_module(name=f'Activ_{layer + 1}', module=ReLU())

		net.add_module(name='Output', module=Linear(self.hparams.num_hidden, self.hparams.in_features, bias=bias))

		net.hparams = self.hparams

		for module in net.modules():
			module.apply(lambda x: apply_rescaling(x, scale=1.))

		self.net = ODEWrapper(net, y_mu=self.y_mu, y_std=self.y_std, dy_mu=self.dy_mu, dy_std=self.dy_std)

		self.integrator = IntegratorWrapper(net=self.net, odeint_str=self.hparams.odeint)

		MD_BiModel.__init__(self, self.hparams)

	def forward(self, t, x):
		assert x.shape[1] == 2 * self.hparams.input_length, f'{x.shape=} {self.hparams.input_length=}'
		assert x.dim() == 3

		x1, x2 = torch.chunk(x, chunks=2, dim=1)
		assert x1.shape[1]==x2.shape[1]==1
		x1 = x1.squeeze(1)
		x2 = x2.squeeze(1)

		assert x1.shape==x2.shape

		x1_for = x1
		x2_back = x2

		total_length = t + 2*self.hparams.input_length
		lin_weights = (torch.arange(0, total_length, dtype=x.dtype, device=x.device)) / (total_length-1)
		assert lin_weights[0]==0 and lin_weights[-1] ==1, f'{lin_weights=}'

		weights = torch.stack([lin_weights.flip([0]), lin_weights], dim=-1)
		weights = weights / weights.sum(dim=-1, keepdim=True)
		weights = weights.unsqueeze(-2)

		'''
		Forwards
		'''
		pred_for = self.integrator(t=t+self.hparams.input_length, x=x1_for)
		assert F.mse_loss(pred_for[:, 0], x[:, 0]) == 0.

		'''
		Backwards
		[q, p] -> [q, -p]
		After integration flip from [x(-1), x(-2), x(-3)] to [x(-3), x(-2), x(-1)] for correct direction
		'''
		# print(x2_back.shape)

		velocity_flip = torch.ones(x2_back.shape[-1], device=x.device)
		velocity_flip[velocity_flip.shape[-1] // 2:] = -1
		x2_back = x2_back * velocity_flip

		pred_back = self.integrator(t=t+self.hparams.input_length, x=x2_back).flip([-2]) # Solve [x_0, x_1, x_2 ... x_T] but flip it for [x_T, ..., x_2, x_1, x_0]
		pred_back = pred_back * velocity_flip # flip velocity since it should face 'forward' in time

		assert F.mse_loss(pred_back[:,-1], x[:,-1])==0., f'{pred_back[0,-1]=} {x[0,-1]=}'

		pred = torch.stack([pred_for, pred_back], dim=-1)
		weights = weights.expand_as(pred)
		out = torch.sum(pred * weights, dim=-1)

		assert out.shape==(x.shape[0], total_length, x.shape[-1])

		return out

class MD_ODE(Module, MD_Model):

	def __init__(self, hparams, scaling ):

		Module.__init__(self)

		self.hparams = hparams

		bias = True
		net = Sequential()
		net.add_module(name='Layer_0', module=Linear(self.hparams.in_features, self.hparams.num_hidden, bias=bias))
		net.add_module(name='Activ_0', module=ReLU())

		for layer in range(self.hparams.num_layers):
			net.add_module(name=f'Layer_{layer+1}', module=Linear(self.hparams.num_hidden, self.hparams.num_hidden, bias=bias))
			net.add_module(name=f'Activ_{layer+1}', module=ReLU())

		net.add_module(name='Output', module=Linear(self.hparams.num_hidden, self.hparams.in_features, bias=bias))

		net.hparams = self.hparams

		for module in net.modules():
			module.apply(lambda x: apply_rescaling(x,scale=1.))

		self.y_mu = scaling['y_mu']
		self.y_std = scaling['y_std']
		self.dy_mu = scaling['dy_mu']
		self.dy_std = scaling['dy_std']

		self.net = ODEWrapper(net, y_mu=self.y_mu, y_std=self.y_std, dy_mu=self.dy_mu, dy_std=self.dy_std)

		self.integrator = IntegratorWrapper(net=self.net, odeint_str = self.hparams.odeint)
		MD_Model.__init__(self, self.hparams)

	def forward(self, t, x):
		out = self.integrator(t=t, x=x) # prepends the t dimension [t, batch_size, features]
		# print(f"{x[0]=} {out[0]=}")
		# exit('@MD_ODENet forward')
		return out

class MD_ODE2Net(torch.nn.Module):

	def __init__(self, in_features, hparams, y_mu=None, y_std=None, dy_mu=None, dy_std=None, std=None):

		super().__init__()

		net = Sequential()
		# net.add_module(name='BN_0', 	module=torch.nn.BatchNorm1d(in_features))
		net.add_module(name='Layer_0', module=Linear(in_features, hparams.num_hidden))
		net.add_module(name='Activ_0', module=Tanh())

		for layer in range(hparams.num_layers):
			# self.net.add_module(name=f'DropOut_{layer+1}', module=Dropout(p=0.2))
			net.add_module(name=f'Layer_{layer+1}', module=Linear(hparams.num_hidden, hparams.num_hidden))
			# self.net.add_module(name=f'Batchnorm_{layer+1}' , module=BatchNorm1d(self.num_hidden))
			net.add_module(name=f'Activ_{layer+1}', module=Tanh())

		net.add_module(name='Output', module=Linear(hparams.num_hidden, in_features))

		# for module in net.modules():
		# 	module.apply(lambda x: apply_rescaling(x,scale=1.))

		self.net = ODE2Wrapper(net, y_mu=y_mu, y_std=y_std, dy_mu=dy_mu, dy_std=dy_std)

		self.integrator = IntegratorWrapper(_net=self.net, _odeint_str=hparams.odeint)

	def forward(self, t, x):

		out = self.integrator(t=t, x=x) # prepends the t dimension [t, batch_size, features]

		return out

class MD_Hamiltonian(Module, MD_Model):

	def __init__(self, hparams, scaling):

		Module.__init__(self)

		self.hparams = hparams

		actfunc = torch.nn.Tanh
		bias = True
		net = Sequential()
		# net.add_module(name='BN_0', 	module=torch.nn.BatchNorm1d(in_features))
		net.add_module(name='Layer_0', module=Linear(self.hparams.in_features, self.hparams.num_hidden))
		net.add_module(name='Activ_0', module=Tanh())

		for layer in range(self.hparams.num_layers):
			# self.net.add_module(name=f'DropOut_{layer+1}', module=Dropout(p=0.2))
			net.add_module(name=f'Layer_{layer + 1}', module=Linear(self.hparams.num_hidden, self.hparams.num_hidden))
			# self.net.add_module(name=f'Batchnorm_{layer+1}' , module=BatchNorm1d(self.num_hidden))
			net.add_module(name=f'Activ_{layer + 1}', module=Tanh())

		net.add_module(name='Output', module=Linear(self.hparams.num_hidden, 1))

		# for module in net.modules():
		# 	module.apply(lambda x: apply_rescaling(x, scale=1.))

		self.net = HamiltonianWrapper(net=net)
		self.integrator = IntegratorWrapper(self.net, odeint_str=self.hparams.odeint)
		MD_Model.__init__(self, self.hparams)

	def forward(self, t, x):

		if x.dim()==3 and x.shape[1]==1:
			x = x.squeeze(1)

		out = self.integrator(x=x, t=t) # prepends the t dimension [t, batch_size, features]

		return out

class MD_LSTM(Module, MD_Model):

	def __init__(self, hparams, scaling):

		Module.__init__(self)

		self.hparams = hparams

		self.lstm = LSTM(input_size=self.hparams.in_features, hidden_size=hparams.num_hidden, num_layers=hparams.num_layers-1, batch_first=True, dropout=0.2)
		self.out_emb = Linear(self.hparams.num_hidden, self.hparams.in_features, bias=True)

		self.y_mu = scaling['y_mu']
		self.y_std = scaling['y_std']
		self.dy_mu = scaling['dy_mu']
		self.dy_std = scaling['dy_std']

		MD_Model.__init__(self, hparams)

	def forward(self, t, x):

		if x.dim() == 2: x = x.unsqueeze(1)
		assert x.dim() == 3

		mode = ['diff', 'autoreg'][0]

		pred, (h,c) = self.lstm(x)
		dx = self.out_emb(pred)

		out = torch.cat([x[:, :1, :], x + dx], dim=1) if mode=='diff' else torch.cat([x[:, :1, :], dx], dim=1)
		for step in range(t - 1):  # because we add the first entry y0 at the beginning
			pred_t, (h,c) = self.lstm(out[:, -1:], (h,c))
			dx_t = self.out_emb(pred_t)

			if mode=='diff':
				if self.dy_std is not None:
					dx_t = self.dy_std * dx_t + self.dy_mu

				out = torch.cat([out, out[:, -1:, :] + dx_t], dim=1)
			else:
				out = torch.cat([out, dx_t], dim=1)

		return out

class MD_RNN(Module, MD_Model):

	def __init__(self, hparams, scaling):

		Module.__init__(self)

		self.hparams = hparams

		self.rnn = RNN(input_size=self.hparams.in_features, hidden_size=self.hparams.num_hidden, num_layers=self.hparams.num_layers-1, batch_first=True)
		self.out_emb = Linear(self.hparams.num_hidden, self.hparams.in_features, bias=True)

		self.y_mu 	= scaling['y_mu']
		self.y_std 	= scaling['y_std']
		self.dy_mu 	= scaling['dy_mu']
		self.dy_std 	= scaling['dy_std']

		MD_Model.__init__(self, hparams)

	def forward(self, t, x):
		''''
		t=3: predition timesteps
		Input: [y0 y1 y2 y3]
		Input Prediction: [y0 y1 y2 y3] + [dy0 dy1 dy2 dy3] -> [y1' y2' y3' y4']
		Output Prediction [y4'] -> AR(t=3-1) -> [y5' y6'] ([y4'] -> [y5'] was already a prediction step)
		Prediction: [y0 y1' y2' y3' y4' | y5' y6' y7']

		'''
		if x.dim() == 2: x = x.unsqueeze(1)
		assert x.dim() == 3

		pred, h = self.rnn(x)
		dx = self.out_emb(pred)
		out = torch.cat([x[:, :1, :], x + dx], dim=1)
		for step in range(t-1):  # because we add the first entry y0 at the beginning
			pred_t, h = self.rnn(out[:, -1:], h)
			dx_t = self.out_emb(pred_t)
			if self.dy_std is not None:
				dx_t = self.dy_std * dx_t + self.dy_mu

			out = torch.cat([out, out[:, -1:, :] + dx_t], dim=1)

		'''
		out = [y1' y2' y3' y4' | y5' y6' y7'] -> [y0 y1' y2' y3' y4' | y5' y6' y7']
		'''

		return out

class MD_UniversalDiffEq(torch.nn.Module):

	class UniversalDiffEq(torch.nn.Module):

		def __init__(self, in_features, embedding_features, hidden_features):

			super().__init__()



			actfunc = torch.nn.Tanh

			# self.embedding_in = Sequential(Linear(in_features, embedding_features))

			self.ode = Sequential(Linear(in_features, hidden_features),
					      actfunc(),
					      Linear(hidden_features, hidden_features),
					      actfunc(),
					      Linear(hidden_features, in_features))

			self.firstpd = FirstOrderPDWrapper(Sequential(	Linear(in_features, hidden_features),
									  actfunc(),
									  Linear(hidden_features, hidden_features),
									  actfunc(),
									  Linear(hidden_features, 1)))

			self.secondpd = SecOrderPDWrapper(Sequential(	Linear(in_features, hidden_features),
									     actfunc(),
									     Linear(hidden_features, hidden_features),
									     actfunc(),
									     Linear(hidden_features, 1)))

		def forward(self, t, _x):

			ode = self.ode(_x)
			firstpd = self.firstpd(t, _x)
			secondpd = self.secondpd(t, _x)

			return ode + firstpd + secondpd

	def __init__(self, in_features, hparams):

		super().__init__()

		net = self.UniversalDiffEq(in_features, embedding_features=0, hidden_features=100)

		self.integrator = IntegratorWrapper(net)

	def forward(self, _t, _x):

		# print(f'{_t=} {_x.shape=}')
		out = self.integrator(_x=_x, _t=_t) # prepends the t dimension [t, batch_size, features]

		return out