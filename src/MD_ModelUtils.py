import future, sys, os, datetime, argparse, copy
# print(os.path.dirname(sys.executable))
import torch
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
# matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch
import torch.optim
from torch import tanh, sigmoid, relu, optim
from torch.nn import Sequential, Conv1d, ConvTranspose1d, Conv2d, ConvTranspose2d, LayerNorm, Linear, BatchNorm1d, UpsamplingBilinear2d
from torch.nn import ReLU, LeakyReLU, Tanh, CELU, Softplus, Sigmoid
from torch.utils.data import DataLoader, TensorDataset
import torch.nn.functional as F


from torchdiffeq import odeint_adjoint, odeint
from torchdiffeq import odeint_adjoint as odeint

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

def apply_weightnorm(_module):
	# print(type(_module))
	if type(_module) in [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear]:
		print(type(_module))

def apply_rescaling(_module, scale=0.1):
	if type(_module) in [torch.nn.Conv1d, torch.nn.Conv2d, torch.nn.Linear]:
		_module.weight.data.mul_(scale)
		if _module.bias is not None:
			_module.bias.data.mul_(scale)


class ForwardPreHook():

	def __init__(self, module, name, hparams):
		'''
		register_forward_hook: executes hook_fn after forward pass of module
		register_forward_pre_hook: executes hook_fn before forward pass of module
		'''

		self.hparams = hparams
		self.name = name
		self.hook = module.register_forward_pre_hook(self.forward_histogram_hook)

	def forward_histogram_hook(self, module, input):
		'''

		:param module:
		:param input:
		:param output:
		:return:
		'''

		# if self.hparams.plot:
		# 	plt.hist(output[0].flatten(), bins=100, alpha=0.5, density=True, label=self.name)
			# plt.show()
		# print(f'@ForwardPreHook{input[0].shape=}')
		# print(f'@ForwardPreHook{input[1].shape=}')
		if self.hparams.log:
			self.hparams.logger.add_histogram(tag=self.name, values=input[0])

class ForwardHook():

	def __init__(self, module, name, hparams):
		'''
		register_forward_hook: executes hook_fn after forward pass of module
		register_forward_pre_hook: executes hook_fn before forward pass of module
		'''

		self.hparams = hparams
		self.name = name
		self.hook = module.register_forward_hook(self.forward_histogram_hook)

	def forward_histogram_hook(self, module, input, output):
		'''

		:param module:
		:param input:
		:param output:
		:return:
		'''

		# if self.hparams.plot:
		# 	plt.hist(output[0].flatten(), bins=100, alpha=0.5, density=True, label=self.name)
		# 	plt.show()
		if self.hparams.log:
			self.hparams.logger.add_histogram(tag=self.name, values=input[0])
class Hook():
	def __init__(self, module, backward=False):
		if backward==False:
			self.hook = module.register_forward_hook(self.hook_fn)
		else:
			self.hook = module.register_backward_hook(self.hook_fn)
	def hook_fn(self, module, input, output):
		self.input = input
		self.output = output
	def close(self):
		self.hook.remove()

class SecOrderPDWrapper(torch.nn.Module):

	# def __init__(self, _input_channels, _hidden_channels, _layers):
	def __init__(self, _net):

		super().__init__()

		self.net = _net
		self.nfe = 0

		self.ddfddx = None


	def forward(self, t, _x):
		self.nfe += 1
		with torch.set_grad_enabled(True):

			_x.requires_grad_()

			f = self.net(_x).sum(dim=-1, keepdim=True)

			dfdx, = torch.autograd.grad(f, inputs=_x, only_inputs=True, create_graph=True, grad_outputs=torch.ones_like(f))

			ddfddx, = torch.autograd.grad(dfdx, inputs=_x, create_graph=True, grad_outputs=torch.ones_like(dfdx))

		self.ddfddx = ddfddx.detach()

		return ddfddx

class FirstOrderPDWrapper(torch.nn.Module):

	# def __init__(self, _input_channels, _hidden_channels, _layers):
	def __init__(self, _net, _mu=None, _std=None):

		super().__init__()

		self.net = _net
		self.mu = _mu
		self.std = _std
		self.nfe = 0

		self.dfdx = None


	def forward(self, t, _x):
		self.nfe += 1

		with torch.set_grad_enabled(True):

			_x.requires_grad_()

			f = self.net(_x)#.sum(dim=-1, keepdim=True)

			dydx, = torch.autograd.grad(f, inputs=_x, create_graph=True, grad_outputs=torch.ones_like(f))

		# print('@FirstOrderPDE.forward')
		# print(f'{t.shape=}, {_x.shape=}, {dfdx.shape=}')
		if self.std is not None and self.mu is not None:
			dydx = self.std * dydx + self.mu
		self.dydx = dydx

		return dydx

class HamiltonianWrapper(torch.nn.Module):

	def __init__(self, net, mu=None, std=None, dir='forward'):

		super().__init__()

		self.dir = dir
		self.net = net
		self.mu = mu
		self.std = std
		self.nfe = 0

	def forward(self, t, x, dir='forward'):
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

		with torch.set_grad_enabled(True):
			x = x.requires_grad_(True)
			# print(f'{x.shape=}')
			H = self.net(x).sum(dim=-1, keepdim=True) # sum the predictions to [batch_size, 1]
			assert H.shape==(H.shape[0],1), f'{H.shape=} {x.shape=}'
			dHdx, = torch.autograd.grad(H, inputs=x, grad_outputs=torch.ones_like(H), create_graph=True)#, grad_outputs=torch.ones_like(f))

		# dfdx = dfdx[:,[2,3,0,1]]
		dHdpos, dHdvel = torch.chunk(dHdx, chunks=2, dim=-1)
		dHdx = torch.cat([dHdvel, -dHdpos], dim=-1)

		if self.dir=='backward':
			# print('@Hamiltonian Wrapper Forward')
			dHdx = - dHdx

		if self.std is not None and self.mu is not None:
			dHdx = self.std * dHdx + self.mu
		# dfdx *=Tensor([1,1,-1,-1]).expand_as(dfdx)

		return dHdx

class ODEWrapper(torch.nn.Module):

	def __init__(self, net, y_mu=None, y_std=None, dy_mu=None, dy_std=None):

		super().__init__()

		self.net 	= net
		self.y_mu 	= y_mu
		self.y_std	= y_std
		self.dy_mu 	= dy_mu
		self.dy_std	= dy_std
		self.nfe 	= 0
		self.dir 	= 'forward'

	def forward(self, t, x):
		self.nfe += 1

		if self.y_std is not None and self.y_mu is not None:
			x = (x-self.y_mu)/self.y_std

		self.input = x
		dy = self.net(x)
		self.output = dy

		if self.dy_std is not None and self.dy_mu is not None:
			dy = self.dy_std * dy + self.dy_mu

		if self.dir == 'backward':
			dy = - dy

		return dy

class ODE2Wrapper(torch.nn.Module):

	def __init__(self, net, y_mu=None, y_std=None, dy_mu=None, dy_std=None, mode='diffeq'):

		super().__init__()

		self.mode = mode

		self.net = net
		self.y_mu = y_mu
		self.y_std = y_std
		self.dy_mu = dy_mu
		self.dy_std = dy_std

		self.dv_mu = dy_mu.chunk(chunks=2, dim=-1)[1]
		self.dv_std = dy_std.chunk(chunks=2, dim=-1)[1]

		self.nfe = 0

		assert mode in ['partial', 'diffeq']
		if mode in ['ode']:
			if net[-1].weight.shape[0]!=(net[0].weight.shape[1]//2):
				print(f'{net[-1].weight.shape[0]}!={(net[0].weight.shape[1]//2)}')
				assert net[-1].weight.shape[0]==(net[0].weight.shape[0]//2), f'Input is {net[0].weight.shape[1]=} -> Output should be {net[0].weight.shape[1]//2} not R^{net[-1].weight.shape[0]=}'

	def forward(self, t, x):

		self.nfe += 1

		s, v = torch.chunk(x, chunks=2, dim=-1)

		if self.y_std is not None and self.y_mu is not None:
			x = (x - self.y_mu)/(self.y_std + 1e-3)

		self.input = x.detach()

		if self.mode=='partial':
			with torch.set_grad_enabled(True):
				x = x.requires_grad_(True)
				v = self.net(x)
				dxdt, = torch.autograd.grad(v, x, create_graph=True, grad_outputs=torch.ones_like(v))
				_, dvdt = torch.chunk(dxdt, chunks=2, dim=-1)
		elif self.mode=='diffeq':
			dy = self.net(x)

		self.output = dy.detach()


		if self.dy_std is not None and self.dy_mu is not None:
			dy = self.dy_std * dy + self.dy_mu

		dv = dy.chunk(chunks=2, dim=-1)[1]


		dy = torch.cat([v, dv], dim=-1)

		# print(f'{dydt.shape=}')
		# exit()

		return dy

class SDEWrapper(torch.nn.Module):

	# def __init__(self, _input_channels, _hidden_channels, _layers):
	def __init__(self, net, mu=None, std=None):

		super().__init__()

		self.model = net
		self.mu = mu
		self.std = std
		self.nfe = 0

	def forward(self, t, _x):
		self.nfe += 1

		p_dydt = self.model(_x)

		dydt = p_dydt.rsample()
		# dydt = p_dydt.loc

		if self.std is not None and self.mu is not None:
			dydt = self.std * dydt + self.mu

		# print('@ODE.forward')
		# print(f'{t.shape=}, {_x.shape=}, {dydt.shape=}')
		return dydt

class IntegratorWrapper(torch.nn.Module):

	def __init__(self, net, odeint_str='rk4'):

		super().__init__()

		self.odeintegator = odeint_str
		self.diffeq = net

	def forward(self, t, x):

		'''
		t = torch.arange(t) creates a vector of [0 1 ... t-1] of length t
		'''
		# print(f"@integratorwrapper 0 {t=}")

		if t==1:
			# self.integration_time = torch.arange(0, _t).type_as(_x)
			self.integration_time = torch.tensor([0, 1]).type_as(x)
		elif t>1:
			t = copy.deepcopy(t) + 1
			self.integration_time = torch.arange(0, t).type_as(x)
		else:
			print('Integration time is negative!')
			exit()

		# print(f"@integratorwrapper 1 {t=}")

		# print('@Integrator.forward')
		# print(f'{_x.shape=}, {_t.shape=}')
		out = odeint_adjoint(self.diffeq, x, self.integration_time, method=self.odeintegator)

		# print(f"@integratorwrapper {t=} {self.integration_time=} {out.shape=}")

		return out.transpose(0,1) # shape=[ timesteps, batch, features] -> shape=[batch, timesteps, features]

	@property
	def nfe(self):
		return self.odefunc.nfe

	@nfe.setter
	def nfe(self, value):
		self.odefunc.nfe = value