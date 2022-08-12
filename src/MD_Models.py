import sys, os, warnings
import numbers, math
from numbers import Number
from typing import Union
import numpy as np

import matplotlib

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch
from torch.nn import functional as F
from torch.nn import Sequential, Module
from torch.nn import Linear, LSTM, RNN, Dropout
from torch.nn import Tanh, LeakyReLU, ReLU
from torch.nn import TransformerEncoder, TransformerEncoderLayer
from torchdyn.models import NeuralODE

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
patch_typeguard()

Tensor = torch.FloatTensor

sys.path.append("/".join(os.getcwd().split("/")[:-1])) # experiments -> MLMD
sys.path.append("/".join(os.getcwd().split("/")[:-2])) # experiments -> MLMD -> PhD

from MLMD.src.MD_ModelUtils import IntegratorWrapper
from MLMD.src.MD_ModelUtils import ODEWrapper, ODE2Wrapper, FirstOrderPDWrapper, SecOrderPDWrapper, HamiltonianWrapper
from MLMD.src.MD_ModelUtils import apply_rescaling, ForwardHook, ForwardPreHook

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# Unidirectional Models
class MD_Model(Module):
	'''
	Parent Class that standardizes integration, forward and loss function (criterion) interface
	'''

	def __init__(self, hparams):

		super().__init__()

		self.hparams = hparams

	def integration(self, t: Number, x_init: Tensor, direction: Union[Number, torch.FloatTensor, torch.Tensor]= 1):
		'''
		Abstraction for the specific integration procedure of the subclassed integrators (ODE, HNN, LSTM etc)
		'''
		raise NotImplementedError

	def __call__(self, t: Number, x: Tensor):
		return self.integration(t=t, x_init=x, direction=1)

	@typechecked
	@torch.no_grad()
	def forecast(self, T: Number, x0: Union[TensorType['BS', -1, 'F'], TensorType[-1, 'F']], t0: Number =None):
		assert (x0.dim()==2 or x0.dim()==3)
		if x0.dim()==2: x0 = x0.unsqueeze(0)
		assert x0.dim()==3, f"{x0.dim()=}"
		assert x0.shape[0]==1, f"{x0.shape=}"

		BS, input_length, Feat = x0.shape

		assert hasattr(self, 'data_mean') and hasattr(self, 'data_std')
		x0 = (x0 - self.data_mean)/self.data_std
		pred_for = self.integration(t=T-input_length, x_init=x0, direction=1).squeeze(0)
		pred_for = pred_for * self.data_std + self.data_mean
		assert pred_for.shape[0]==T and pred_for.dim()==2, f"{pred_for.shape=} != {T}"
		return pred_for

	@typechecked
	def set_diffeq_output_scaling_statistics(self, dy_mu: TensorType[1, 'F'], dy_std: TensorType[1, 'F']):
		'''
		We can't set the scaling during intitialization as it interferes with the logging pipeline (can't log an entire tensor)
		@param dy_mu:
		@param dy_std:
		@return:
		'''

		raise NotImplementedError

	def normalize_data(self, data):
		assert hasattr(self, 'data_mean') and hasattr(self, 'data_std')
		return (data - self.data_mean)/self.data_std

	def unnormalize_data(self, data):
		assert hasattr(self, 'data_mean') and hasattr(self, 'data_std')
		return data * self.data_std + self.data_mean

	def criterion(self, _pred, _target, forecasting=True):

		assert _pred.shape==_target.shape, f' {_pred.shape=} VS {_target.shape=}'
		assert _pred.dim()==_target.dim()==2 or _pred.dim() == _target.dim() == 3, f"{_pred.shape=}, {_target.shape=} != [Num_Traj, T, F]"
		if _pred.dim()==3:
			assert torch.sum(_pred[:,0]- _target[:, 0])<=1e-6, f" y0's aren't the same"
		elif _pred.dim()==2:
			assert torch.sum(_pred[0] - _target[0])<=1e-6, f" y0's aren't the same"

		if self.hparams.criterion=='MSE':
			return F.mse_loss(_pred, _target)
		elif self.hparams.criterion=='MAE':
			return (_pred -_target).abs().mean() #+ (self.unnormalize_data(_pred).pow(2).sum(dim=-1) - 1).abs().mean()
		else:
			raise Exception(f'Wrong Criterion chosen: {self.hparams.criterion}')

class FeedForwardNetwork(Module):

	'''
	Separate Class for Feed Forward Neural Networks for which we can set the integration direction in or before the forward method
	'''

	def __init__(self, hparams):

		super(FeedForwardNetwork, self).__init__()
		self.hparams = hparams

		bias = True

		self.net = Sequential()
		self.net.add_module(name='Layer_0', module=Linear(hparams.in_features, hparams.num_hidden, bias=bias))
		self.net.add_module(name='Activ_0', module=Tanh())

		for layer in range(hparams.num_layers):
			self.net.add_module(name=f'Layer_{layer + 1}', module=Linear(hparams.num_hidden, hparams.num_hidden, bias=bias))
			self.net.add_module(name=f'Activ_{layer + 1}', module=Tanh())

		self.net.add_module(name='Output', module=Linear(hparams.num_hidden, hparams.out_features, bias=bias))

		for module in self.net.modules():
			module.apply(lambda x: apply_rescaling(x, scale=1.))

		self.integration_direction = 1

	def forward(self, x):
		if type(self.integration_direction) is numbers.Number:
			assert (self.integration_direction in [1, -1])
		if type(self.integration_direction) is torch.Tensor:
			assert self.integration_direction.shape==torch.Size([x.shape[0],1,1])

		BS, _, F = x.shape

		dx = self.integration_direction * self.net(x)
		if self.hparams.diffeq_output_scaling: dx = dx * self.dy_std + self.dy_mu

		assert dx.shape==(BS, 1, F)

		return dx

class MD_VAR(MD_Model):

	'''
	Data: 	[BS, T_in, F]
	Params: [1, 1, T_in]
	Out:	[1, 1, F]
	Out[BS, 1, F] = Params[1, 1, T_in] @ Data[BS, T_in, F]

	PyTorch Doc bmm(): (b x n x m) @ (b x m x p) = (b x n x p ) https://pytorch.org/docs/stable/generated/torch.bmm.html?highlight=bmm#torch.bmm

	'''

	def __init__(self, hparams):

		super().__init__(hparams)


		self.params = torch.nn.Parameter(torch.randn(1, 1, self.hparams.input_length)/(self.hparams.input_length*self.hparams.in_features))

	@typechecked
	def set_diffeq_output_scaling_statistics(self, dy_mu: TensorType[1, 'F'], dy_std: TensorType[1, 'F']):
		'''
		We can't set the scaling during intitialization as it interferes with the logging pipeline (can't log an entire tensor)
		@param dy_mu:
		@param dy_std:
		@return:
		'''

		self.dy_mu 	= torch.nn.Parameter(dy_mu, requires_grad=False)
		self .dy_std 	= torch.nn.Parameter(dy_std, requires_grad=False)

	def integration(self, t: Number, x_init: Tensor, direction: Union[Number, torch.FloatTensor, torch.Tensor]= 1):
		'''

		@param t:
		@param x_init: shape=[BS, T, F]
		@param direction:
		@return:
		[b, n, p] = torch.bmm( [b, n, m], [b, m, p] )
		'''
		BS, T_in, F = x_init.shape
		out = torch.cat([x_init, x_init[:,-1:] + torch.matmul( self.params, x_init)], dim=1)
		for t_ in range(t-1):
			out = torch.cat([out, out[:, -1:] + torch.matmul(self.params, out[:,-self.hparams.input_length:])], dim=1)

		assert out.shape[1]==T_in+t, f'{out.shape=} VS T_total={x_init.shape[1] + t}'
		return out

class MD_ODE_SecOrder(MD_Model):

	def __init__(self, hparams, scaling=None):

		MD_Model.__init__(self, hparams)

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

		if scaling is not None:
			self.y_mu = scaling['y_mu']
			self.y_std = scaling['y_std']
			self.dy_mu = scaling['dy_mu']
			self.dy_std = scaling['dy_std']

		self.net = NeuralODE(func=net, order=2, sensitivity='adjoint', solver='euler')


	def integration(self, t, x):
		'''
		x.shape = [BS, InputLength, F]
		out.shape = [BS, OutputLength, F]
		'''
		assert x.dim() == 3
		assert x.shape[1] == 1

		out = self.net.trajectory(x, torch.linspace(0, 1, t + 1))  # shape=[T, BS, 1, F]
		out = out.squeeze(-2).permute(1, 0, 2)  # shape: [T, BS, 1, F] -> squeeze(-2) -> [T, BS, F] -> [BS, T, F]

		return out

class MD_ODE(MD_Model):

	def __init__(self, hparams, scaling=None):
		MD_Model.__init__(self, hparams)

		self.net = FeedForwardNetwork(hparams)

		self.ode = NeuralODE(self.net, order=1, sensitivity='adjoint', solver='euler')
		self.net.dy_mu = None
		self.net.dy_std =None

	@typechecked
	def set_diffeq_output_scaling_statistics(self, dy_mu: TensorType[1, 'F'], dy_std: TensorType[1, 'F']):
		'''
		We can't set the scaling during intitialization as it interferes with the logging pipeline (can't log an entire tensor)
		@param dy_mu:
		@param dy_std:
		@return:
		'''

		self.net.dy_mu = torch.nn.Parameter(dy_mu, requires_grad=False)
		self.net.dy_std = torch.nn.Parameter(dy_std, requires_grad=False)

	@typechecked
	def integration(self, t, x_init: TensorType['BS', 'T_init', 'F'], x_final: Union[TensorType['BS', 'T_final', 'F'], None] = None, direction: Union[TensorType['BS', 1, 1], Number] = 1, diff_order=1):
		'''
		x.shape = [BS, InputLength, F]
		out.shape = [BS, OutputLength, F]
		'''
		assert x_init.dim() == 3
		assert x_init.shape[1] == 1
		# assert direction in [1, -1]

		self.ode.defunc.m.integration_direction = direction
		self.ode.defunc.m.diff_order = diff_order

		if self.hparams.diffeq_output_scaling: assert hasattr(self.ode.defunc.m, 'dy_mu'), f"DiffEq output scaling enabled but no scaling statistics detected"

		out = self.ode.trajectory(x_init, torch.linspace(0, t + 1, t + 1).to(x_init.device))  # shape=[T, BS, 1, F]
		out = out.squeeze(-2).permute(1, 0, 2)  # shape: [T, BS, 1, F] -> squeeze(-2) -> [T, BS, F] -> [BS, T, F]
		self.out = out
		return out

class MD_Hamiltonian(MD_Model):

	class HNNWrapper(torch.nn.Module):
		def __init__(self, net: torch.nn.Module):
			super().__init__()

			self.net = net
			self.integration_direction = None

		def forward(self, x):
			'''
			q -> "position"
			p -> "momentum"
			dq/dt = dH/dp
			dp/dt = -dH/dq

			dfdx is [batch_size, dH/dq+dH/dp]
			:param t:
			:param x: [numbatchsize, q+p]
			:return:
			'''

			# assert self.integration_direction in [1,-1], f"HNNWrapper Integration Direction is {self.integration_direction}"

			with torch.set_grad_enabled(True):
				x = x.requires_grad_(True)
				H = self.net(x).sum(dim=-1, keepdim=True)  # sum the predictions to [batch_size, 1]

				dHdx, = torch.autograd.grad(H, inputs=x, grad_outputs=torch.ones_like(H),
							    create_graph=True)  # , grad_outputs=torch.ones_like(f))

			dHdpos, dHdvel = torch.chunk(dHdx, chunks=2, dim=-1)
			dHdx = self.integration_direction * torch.cat([dHdvel, -dHdpos], dim=-1)

			return dHdx

	def __init__(self, hparams):
		MD_Model.__init__(self, hparams)

		net = FeedForwardNetwork(hparams)

		self.hamiltonian = NeuralODE(func=self.HNNWrapper(net), order=1, sensitivity='adjoint', solver='euler')
		self.hamiltonian.defunc.m.net.dy_mu = None
		self.hamiltonian.defunc.m.net.dy_std = None

	@typechecked
	def set_diffeq_output_scaling_statistics(self, dy_mu: TensorType[1, 'F'], dy_std: TensorType[1, 'F']):
		'''
		We can't set the scaling during intitialization as it interferes with the logging pipeline (can't log an entire tensor)
		@param dy_mu:
		@param dy_std:
		@return:
		'''

		self.hamiltonian.defunc.m.net.dy_mu = torch.nn.Parameter(dy_mu, requires_grad=False)
		self.hamiltonian.defunc.m.net.dy_std = torch.nn.Parameter(dy_std, requires_grad=False)

	@typechecked
	def integration(self, t, x_init: TensorType['BS', 'T_init', 'F'], x_final: Union[TensorType['BS', 'T_final', 'F'], None] = None, direction: Union[TensorType['BS', 1, 1], Number] = 1, diff_order=1):
		'''
		x.shape = [BS, InputLength, F]
		out.shape = [BS, OutputLength, F]
		'''
		assert x_init.dim() == 3
		assert x_init.shape[1] == 1

		'''defunc.m is the neural network model'''
		self.hamiltonian.defunc.m.integration_direction = direction

		out = self.hamiltonian.trajectory(x_init, torch.linspace(0, 1, t + 1))  # shape=[T, BS, 1, F]
		out = out.squeeze(-2).permute(1, 0, 2)  # shape: [T, BS, 1, F] -> squeeze(-2) -> [T, BS, F] -> [BS, T, F]

		return out

class MD_LSTM(MD_Model):

	def __init__(self, hparams):

		MD_Model.__init__(self, hparams)

		self.lstm = LSTM(input_size=self.hparams.in_features, hidden_size=hparams.num_hidden, num_layers=hparams.num_layers - 1,
				 batch_first=True)
		self.out_emb 	= Linear(self.hparams.num_hidden, self.hparams.in_features, bias=True)
		self.dy_mu 	= None
		self.dy_std 	= None

	@typechecked
	def set_diffeq_output_scaling_statistics(self, dy_mu: TensorType[1, 'F'], dy_std: TensorType[1, 'F']):
		'''
		We can't set the scaling during intitialization as it interferes with the logging pipeline (can't log an entire tensor)
		Using Parameter and requires_grad=None to move it to GPU but disable gradients and gradient descent updates
		@param dy_mu:
		@param dy_std:
		@return:
		'''

		self.dy_mu 	= torch.nn.Parameter(dy_mu, requires_grad=False)
		self.dy_std 	= torch.nn.Parameter(dy_std, requires_grad=False)

	@typechecked
	def integration(self, t, x_init: TensorType['BS', 'T_init', 'F'], direction: Union[TensorType['BS', 1, 1], Number]=1):
		'''
		:param t:
		:param x: [x(0), x(1), x(2)]
		:direction : sign of integration direction, whether to add dx/dt in forward integration or subtract in backward integration
		:return:
		'''
		if x_init.dim() == 2: x_init = x_init.unsqueeze(1)
		assert x_init.dim() == 3
		BS, T, F = x_init.shape
		'''
		[ x1, x2, x3 ] -> LSTM -> [ dx1, dx2, dx3 ], (h, c) -> [ x1, x1+dx1, x1+dx1+dx2, x1+dx1+dx2+dx3 ]
		'''

		pred, (h, c) = self.lstm(x_init)
		dx = self.out_emb(pred)
		if self.hparams.diffeq_output_scaling: dx = dx * self.dy_std + self.dy_mu

		''' 
		Training: 	[x0, x0+dx0, 	x0+dx0+dx1, 	x0+dx0+dx1+dx2 	| Autoregressive Prediction ]
		Validation:	[x0, x1, 	x2, 		x3		| Autoregressive Prediction ] = just loading up the hidden states
		'''
		# if self.training: out = torch.cat([x_init[:, :1], x_init[:, :1] + direction * torch.cumsum(dx, dim=1)], dim=1)
		# elif not self.training: out = torch.cat([x_init, x_init[:, -1:] + direction * dx[:, -1:]], dim=1)
		out = torch.cat([x_init, x_init[:, -1:] + direction * dx[:, -1:]], dim=1)

		assert out.shape==(BS, T+1, F), f"{out.shape=} VS {(BS, T+1, F)}"

		for step in range(t - 1):  # because we add the first entry y0 at the beginning
			pred_t, (h, c) = self.lstm(out[:, -1:], (h, c))
			dx_t = self.out_emb(pred_t)
			if self.hparams.diffeq_output_scaling: dx_t = dx_t * self.dy_std + self.dy_mu
			out = torch.cat([out, out[:, -1:] + direction*dx_t], dim=1)

		assert out.shape==(BS, T+t, F), f"{out.shape=} VS {(BS, T+t, F)}"
		return out

class MD_RNN(MD_Model):

	def __init__(self, hparams, scaling=None):

		MD_Model.__init__(self, hparams)

		self.rnn = RNN(input_size=self.hparams.in_features, hidden_size=self.hparams.num_hidden, num_layers=self.hparams.num_layers - 1,
			       batch_first=True)
		self.out_emb = Linear(self.hparams.num_hidden, self.hparams.in_features, bias=True)

		self.dy_mu = None
		self.dy_std = None

	@typechecked
	def set_diffeq_output_scaling_statistics(self, dy_mu: TensorType[1, 'F'], dy_std: TensorType[1, 'F']):
		'''
		We can't set the scaling during intitialization as it interferes with the logging pipeline (can't log an entire tensor)
		Using Parameter and requires_grad=None to move it to GPU but disable gradients and gradient descent updates
		@param dy_mu:
		@param dy_std:
		@return:
		'''

		self.dy_mu = torch.nn.Parameter(dy_mu, requires_grad=False)
		self.dy_std = torch.nn.Parameter(dy_std, requires_grad=False)

	def integration(self, t, x_init, direction=1):
		''''
		t=3: predition timesteps
		direction : sign of integration direction, whether to add dx/dt in forward integration or subtract in backward integration
		Input: [y0 y1 y2 y3]
		Input Prediction: [y0 y1 y2 y3] + [dy0 dy1 dy2 dy3] -> [y1' y2' y3' y4']
		Output Prediction [y4'] -> AR(t=3-1) -> [y5' y6'] ([y4'] -> [y5'] was already a prediction step)
		Prediction: [y0 y1' y2' y3' y4' | y5' y6' y7']
		'''

		if x_init.dim() == 2: x_init = x_init.unsqueeze(1)
		assert x_init.dim() == 3

		pred, h = self.rnn(x_init)
		dx = self.out_emb(pred)
		if self.hparams.diffeq_output_scaling: dx = dx * self.dy_std + self.dy_mu

		out = torch.cat([x_init, x_init[:, -1:, :] + direction*dx[:, -1:, :]], dim=1)
		for step in range(t - 1):  # because we add the first entry y0 at the beginning
			pred_t, h = self.rnn(out[:, -1:], h)
			dx_t = self.out_emb(pred_t)
			if self.hparams.diffeq_output_scaling: dx = dx * self.dy_std + self.dy_mu
			out = torch.cat([out, out[:, -1:, :] + direction*dx_t], dim=1)

		'''
		out = [y1' y2' y3' y4' | y5' y6' y7'] -> [y0 y1' y2' y3' y4' | y5' y6' y7']
		'''
		return out

# Bidirectional Models
class MD_BiModel(Module):

	def __init__(self, hparams):

		super(MD_BiModel, self).__init__()

		self.hparams = hparams

		if self.hparams.interpolation=='transformer':
			self.interpolation_transformer = Interpolation_Transformer(hparams)

	@typechecked
	def set_diffeq_output_scaling_statistics(self, dy_mu: TensorType[1, 'F'], dy_std: TensorType[1, 'F']):
		'''
		We can't set the scaling during intitialization as it interferes with the logging pipeline (can't log an entire tensor)
		We're essentially passing on the arguments to the uni-directional instantiation of the model
		@param dy_mu:
		@param dy_std:
		@return:
		'''

		self.mlmd_model.set_diffeq_output_scaling_statistics(dy_mu, dy_std)

	def criterion(self, _pred, _target, forecasting=False, validate_args=True):
		assert _pred.shape == _target.shape, f'{_pred.shape=} {_target.shape=}'

		assert _pred.shape == _target.shape, f' {_pred.shape=} VS {_target.shape=}'
		assert _pred.dim() == _target.dim() == 2 or _pred.dim() == _target.dim() == 3, f"{_pred.shape=}, {_target.shape=} != [Num_Traj, T, F]"
		if _pred.dim() == 3 and validate_args:
				assert torch.sum(_pred[:, 0] - _target[:, 0]) <= 1e-4, f" y0's aren't the same"
				if not forecasting: assert (_pred[:, -1] - _target[:, -1]).abs().sum() <= 1e-4, f" yT's aren't the same"
		elif _pred.dim() == 2 and validate_args:
				assert (_pred[0] - _target[0]).abs().sum() <= 1e-4, f" y0's aren't the same"
				if not forecasting: assert (_pred[-1] - _target[-1]).abs().sum() <= 1e-4, f" yT's aren't the same"

		if self.hparams.criterion == 'MSE':
			return F.mse_loss(_pred, _target)
		elif self.hparams.criterion == 'MAE':
			return (_pred - _target).abs().mean()
		else:
			raise Exception(f'Wrong Criterion chosen: {self.hparams.criterion}')

	def __call__(self, t, x):
		assert x.shape[1] == 2 * self.hparams.input_length
		assert x.dim() == 3

		x_init, x_final = torch.chunk(x, chunks=2, dim=1)
		BS, input_length, Feat = x_init.shape
		output_length = t

		'''Solution to DiffEq'''

		if self.hparams.interpolation and self.hparams.interpolation_mode=='adiabatic':
			self.pred_for, self.pred_back = self.integration(t=input_length+t, x_init=x_init, x_final=x_final, direction='bidirectional')
			assert self.pred_for.shape[1]==self.pred_back.shape[1]==(x.shape[1]+t), f"{self.pred_for.shape=} VS {self.pred_back.shape=} VS t={x.shape[1]+t}"
			if self.hparams.interpolation=='transformer':
				pred = self.transformer_interpolation(self.pred_for, self.pred_back, x_init, x_final, input_length, output_length)
			elif self.hparams.interpolation_mode == 'adiabatic':
				pred = self.adiabatic_convection_interpolation(input_length=input_length, output_length=t, x_init=self.pred_for, x_final=self.pred_back)

		elif not self.hparams.interpolation:
			'''
			We will randomly integrate forward and backward
			'''

			forward = np.random.choice([True,False])
			# forward = False; warnings.warn('Only integrating backwards during Training')
			# forward = True; warnings.warn('Only integrating forward during Training')

			if forward:
				self.pred_for = self.integration(t=input_length + t, x_init=x_init, direction='forward')
				self.pred_for[:,-input_length:,:] = x_final
				pred = self.pred_for
			elif not forward:
				self.pred_back = self.integration(t=input_length + t, x_final=x_final, direction='backward')
				self.pred_back[:,:input_length,:] = x_init
				pred = self.pred_back

		else:
			raise ValueError(f'No output generated due to {self.hparams.interpolation=} and {self.hparams.interpolation_mode}')

		self.pred = pred

		assert pred.shape == (BS, 2 * input_length + t, Feat), f"{pred.shape=} VS {(BS, 2 * input_length + t, Feat)}"
		return pred

	def integration(self, t, x_init=None, x_final=None, direction='forward'):
		'''
		:param t:
		:param x_init: [x(0), x(1), x(2)]
		:return:

		Backward Integration:
			Time reversion requires flipping the input sequence and reveresing the velocities
			[ xN-2, xN-1, xN ] -> flip -> [ xN, xN-1, xN-2 ]
			[ vN-2, vN-1, vN ] -> flip -> [ vN, vN-1, vN-2 ]
			The vels still point 'forward' in time
			Thus we reverse the velocities by multiplying them with -1

		Combinations of integrator/diffeq learning vs system type for backward integration
				Integrator	|	DiffEq
		Dynamic System		Y	|	Y			[has momentum]
		--------------------------------+---------------
		State Vector		N	|	Y			[doesnt have momentum]
				[flip momentum]	   [subtracts time derivative]
		'''

		assert hasattr(self, 'mlmd_model')

		assert direction in ['forward', 'backward', 'bidirectional']
		assert self.hparams.system in ['dynsys', 'statevec']
		if self.hparams.integration_mode=='int': assert self.hparams.system=='dynsys', f"Cant use integration mode with a state vector system, since momentum is missing"
		if direction == 'bidirectional': # integrate both forward and backward at the same time
			BS, Input_Length, F = x_init.shape
			assert x_init.shape == x_final.shape

			if self.hparams.integration_mode=='int': # learning an integrator by reverting the momentum of the backward integration
				''' Flip velocity direction for backward integration '''
				vel_reversion = torch.cat([torch.ones(F // 2, device=device), -torch.ones(F // 2, device=device)], dim=0).reshape(1, 1, -1)
				x_final = vel_reversion * x_final
				direction_ = 1
			elif self.hparams.integration_mode=='diffeq': # learning instantaneous change/time derivative
				direction_ = torch.cat([torch.ones(BS, device=device), -torch.ones(BS, device=device)], dim=0).reshape(-1,1,1)
			else:
				raise AssertionError(f"Integrator mode {self.hparams.integration_mode} incompatible with system {self.hparams.system}")

			'''Stack initial and preprocessed (such that we can integrate forward in time) final condition for a single forward pass'''
			x_final = x_final.flip([1])
			x = torch.cat([x_init, x_final], dim=0)
			pred_for, pred_back = self.mlmd_model.integration(t, x, direction=direction_).chunk(chunks=2, dim=0)
			if self.hparams.integration_mode == 'int': pred_back = vel_reversion * pred_back
			return pred_for, pred_back.flip([1])
		elif direction == 'forward':
			assert type(x_init) == torch.Tensor
			return self.mlmd_model.integration(t, x_init=x_init, direction=1)
		elif direction == 'backward':
			assert x_final is not None
			assert type(x_final) == torch.Tensor
			BS, Input_Length, F = x_final.shape
			if self.hparams.integration_mode == 'int':  # learning an integrator by reverting the momentum of the backward integration
				''' Flip velocity direction for backward integration '''
				vel_reversion = torch.cat([torch.ones(F // 2, device=device), -torch.ones(F // 2, device=device)], dim=0).reshape(1, -1)
				x_final = vel_reversion * x_final
				direction_ = 1
			if self.hparams.integration_mode=='diffeq':
				direction_ = -1
			pred_back = self.mlmd_model.integration(t, x_final.flip([1]), direction=direction_)

			if self.hparams.integration_mode=='int': pred_back = vel_reversion * pred_back

			return pred_back.flip([1])

	@torch.no_grad()
	def forecast(self, T, x0, t0=None):
		assert (x0.dim() == 2 or x0.dim() == 3)
		if x0.dim() == 2: x0 = x0.unsqueeze(0)
		assert x0.dim() == 3, f"{x0.dim()=}"
		assert x0.shape[0] == 1, f"{x0.shape=}"

		BS, input_length, Feat = x0.shape

		assert hasattr(self, 'data_mean') and hasattr(self, 'data_std')
		x0 = (x0 - self.data_mean) / self.data_std
		pred_for = self.integration(t=T - input_length, x_init=x0, direction='forward').squeeze(0)
		pred_for = pred_for * self.data_std + self.data_mean
		assert pred_for.shape[0] == T and pred_for.dim() == 2, f"{pred_for.shape=} != {T}"
		return pred_for

	def adiabatic_convection_interpolation(self, input_length, output_length, x_init, x_final):
		'''
		input_length: scalar
		output_length: scalar
		x: input data to forward pass to extract dtype and device
		For input_length=3 and output_length=3 we have forward_weights=[ 1 1 1 | 0.75, 0.5, 0.25 | 0 0 0 ]
		For input_length=3 and output_length=4 we have forward_weights=[ 1 1 1 | 0.8, 0.6, 0.4 0.2 | 0 0 0 ]
		'''
		forward_weights = torch.linspace(start=1, end=0, steps=output_length + 2, dtype=x_init.dtype, device=x_init.device)[1:-1]
		forward_weights = torch.cat([torch.ones(input_length, dtype=x_init.dtype, device=x_init.device),
					     forward_weights,
					     torch.zeros(input_length, dtype=x_init.dtype, device=x_init.device)])
		backward_weights = forward_weights.flip([0])

		''' [T] -> stack -> [T, 2] -> [1,T,1,2] <=> pred.shape=[BS, T, Feat, 2] '''
		weights = torch.stack([forward_weights, backward_weights], dim=-1).unsqueeze(0).unsqueeze(-2)

		pred = torch.stack([x_init, x_final], dim=-1)

		assert pred.shape[1] == weights.shape[1], f"Time axis not matching: {pred.shape=} VS {weights.shape}"
		out = torch.sum(pred * weights, dim=-1)

		return out

	def constant_interpolation(self, input_length, output_length, x_init, x_final):
		'''
		input_length: scalar
		output_length: scalar
		x: input data to forward pass to extract dtype and device
		For input_length=3 and output_length=3 we have forward_weights=[ 1 1 1 | 0.75, 0.5, 0.25 | 0 0 0 ]
		For input_length=3 and output_length=4 we have forward_weights=[ 1 1 1 | 0.8, 0.6, 0.4 0.2 | 0 0 0 ]
		'''
		forward_weights = 0.5*torch.ones((output_length + 2,), dtype=x_init.dtype, device=x_init.device)[1:-1]
		forward_weights = torch.cat([torch.ones(input_length, dtype=x_init.dtype, device=x_init.device),
					     forward_weights,
					     torch.zeros(input_length, dtype=x_init.dtype, device=x_init.device)])
		backward_weights = forward_weights.flip([0])

		''' [T] -> stack -> [T, 2] -> [1,T,1,2] <=> pred.shape=[BS, T, Feat, 2] '''
		weights = torch.stack([forward_weights, backward_weights], dim=-1).unsqueeze(0).unsqueeze(-2)

		pred = torch.stack([x_init, x_final], dim=-1)

		assert pred.shape[1] == weights.shape[1], f"Time axis not matching: {pred.shape=} VS {weights.shape}"
		out = torch.sum(pred * weights, dim=-1)

		return out

	def transformer_interpolation(self, pred_for, pred_back, x_init, x_final, input_length, output_length):

		trajs = torch.cat([pred_for, pred_back], dim=-1)

		interpolation = self.interpolation_transformer(trajs)
		interpolation = torch.cat([x_init, interpolation[:,input_length:-input_length], x_final], dim=1)

		return interpolation

	def only_forward_integration(self, input_length, output_length, x_init, x_final):

		'''
		input_length: scalar
		output_length: scalar
		x: input data to forward pass to extract dtype and device
		For input_length=3 and output_length=3 we have forward_weights=[ 1 1 1 | 0.75, 0.5, 0.25 | 0 0 0 ]
		For input_length=3 and output_length=4 we have forward_weights=[ 1 1 1 | 0.8, 0.6, 0.4 0.2 | 0 0 0 ]
		'''

		forward_weights = torch.ones((output_length + 2,), dtype=x_init.dtype, device=x_init.device)[1:-1]
		forward_weights = torch.cat([torch.ones(input_length, dtype=x_init.dtype, device=x_init.device),
					     forward_weights,
					     torch.zeros(input_length, dtype=x_init.dtype, device=x_init.device)])
		backward_weights = torch.cat([torch.zeros(input_length, dtype=x_init.dtype, device=x_init.device),
					      torch.zeros((output_length,), dtype=x_init.dtype, device=x_init.device),
					      torch.ones(input_length, dtype=x_init.dtype, device=x_init.device)])

		''' [T] -> stack -> [T, 2] -> [1,T,1,2] <=> pred.shape=[BS, T, Feat, 2] '''
		weights = torch.stack([forward_weights, backward_weights], dim=-1).unsqueeze(0).unsqueeze(-2)

		pred = torch.stack([x_init, x_final], dim=-1)

		assert pred.shape[1] == weights.shape[1], f"Time axis not matching: {pred.shape=} VS {weights.shape}"
		out = torch.sum(pred * weights, dim=-1)

		return out

class MD_BiDirectional_RNN(MD_BiModel):

	def __init__(self, hparams, scaling=None):

		MD_BiModel.__init__(self, hparams)

		self.mlmd_model = MD_RNN(hparams=hparams, scaling=scaling)
		self.mlmd_model.dy_mu = None
		self.mlmd_model.dy_std = None

	# def integration(self, t, x, direction):
	# 	'''
	# 	:param t:
	# 	:param x: [x(0), x(1), x(2)]
	# 	:return:
	# 	'''
		# return self.mlmd_model.forward(t, x, direction)

class MD_BiDirectional_LSTM(MD_BiModel):

	def __init__(self, hparams, scaling=None):

		MD_BiModel.__init__(self, hparams)

		self.mlmd_model = MD_LSTM(hparams=hparams)
		self.mlmd_model.dy_mu = None
		self.mlmd_model.dy_std = None

class MD_BiDirectional_ODE(MD_BiModel):

	def __init__(self, hparams):
		MD_BiModel.__init__(self, hparams)

		self.mlmd_model = MD_ODE(hparams=hparams)

class MD_BiDirectional_Hamiltonian(MD_BiModel):

	def __init__(self, hparams):
		MD_BiModel.__init__(self, hparams)
		self.mlmd_model = MD_Hamiltonian(hparams=hparams)

	# def integration(self, t, x, direction):
	# 	return self.mlmd_model(t, x, direction)  # for whatever reason we have to pass on 't+1'

class Interpolation_Transformer(Module):

	class PositionalEncoding(Module):

		def __init__(self, d_model, dropout=0.1, max_len=5000):
			super().__init__()
			self.dropout = Dropout(p=dropout)

			pe = torch.zeros(max_len, d_model)
			position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
			div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
			pe[:, 0::2] = torch.sin(position * div_term)
			pe[:, 1::2] = torch.cos(position * div_term)
			pe = pe.unsqueeze(0).transpose(0, 1)
			self.register_buffer('pe', pe)


		def forward(self, x):
			x = x + self.pe[:x.size(0), :]
			return self.dropout(x)

	def __init__(self, hparams):

		super().__init__()
		self.hparams = hparams

		ninp = self.hparams.in_features*2 # because we the forward and the backward solution
		nhead = 2
		nhid = hparams.num_hidden
		nlayers = 2
		dropout = 0.5


		self.pos_encoder = self.PositionalEncoding(ninp, dropout)

		encoder_layers = TransformerEncoderLayer(ninp, nhead, nhid, dropout)
		self.transformer_encoder = TransformerEncoder(encoder_layers, nlayers)
		self.final_layer = Linear(ninp, self.hparams.in_features)


	def generate_square_subsequent_mask(self, sz):
		mask = (torch.triu(torch.ones(sz, sz)) == 1).transpose(0, 1)
		mask = mask.float().masked_fill(mask == 0, float('-inf')).masked_fill(mask == 1, float(0.0))
		return mask

	def forward(self, src):

		src = self.pos_encoder(src)
		output = self.transformer_encoder(src)
		output = self.final_layer(output)

		return output


