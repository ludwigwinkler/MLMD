import sys, os

import matplotlib

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch
from torch.nn import functional as F
from torch.nn import Sequential, Module
from torch.nn import Linear, LSTM, RNN
from torch.nn import Tanh, LeakyReLU, ReLU
from torchdyn.models import NeuralDE

from torchdiffeq import odeint_adjoint as odeint

Tensor = torch.Tensor

sys.path.append("/".join(os.getcwd().split("/")[:-1])) # experiments -> MLMD
sys.path.append("/".join(os.getcwd().split("/")[:-2])) # experiments -> MLMD -> PhD

from MLMD.src.MD_ModelUtils import IntegratorWrapper
from MLMD.src.MD_ModelUtils import ODEWrapper, ODE2Wrapper, FirstOrderPDWrapper, SecOrderPDWrapper, HamiltonianWrapper
from MLMD.src.MD_ModelUtils import apply_rescaling, ForwardHook, ForwardPreHook

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


# Unidirectional Models
class MD_Model(Module):

	def __init__(self, hparams):

		super().__init__()

		self.hparams = hparams

	def integration(self, t, x):
		'''
		Abstraction for the specific integration procedure of the subclassed integrators (ODE, HNN, LSTM etc)
		'''
		raise NotImplementedError

	def forward(self, t, x):
		return self.integration(t, x)

	def criterion(self, _pred, _target, mode=None):

		assert _pred.shape==_target.shape, f' {_pred.shape=} VS {_target.shape=}'
		assert torch.sum(_pred[:,:self.hparams.input_length]- _target[:, :self.hparams.input_length])==0, f" y0's aren't the same"

		if mode is None:
			mode = self.hparams.criterion

		if mode=='T':
			return F.mse_loss(_pred[:,-1], _target[:,-1])
		elif mode=='t':
			return F.mse_loss(_pred, _target)
		else:
			raise Exception(f'Wrong Criterion chosen: {self.hparams.criterion}')


class MD_ODE(MD_Model):

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

		self.net = NeuralDE(func=net, order=1, sensitivity='adjoint', solver='euler')


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

		self.net = NeuralDE(func=net, order=2, sensitivity='adjoint', solver='euler')


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

class MD_Hamiltonian(MD_Model):
	class HNN(torch.nn.Module):
		def __init__(self, net: torch.nn.Module):
			super().__init__()

			self.H = net

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

			with torch.set_grad_enabled(True):
				x = x.requires_grad_(True)
				H = self.H(x).sum(dim=-1, keepdim=True)  # sum the predictions to [batch_size, 1]

				dHdx, = torch.autograd.grad(H, inputs=x, grad_outputs=torch.ones_like(H),
							    create_graph=True)  # , grad_outputs=torch.ones_like(f))

			dHdpos, dHdvel = torch.chunk(dHdx, chunks=2, dim=-1)
			dHdx = torch.cat([dHdvel, -dHdpos], dim=-1)

			return dHdx

	def __init__(self, hparams, scaling=None):
		MD_Model.__init__(self, hparams)

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

		self.net = NeuralDE(func=self.HNN(net), order=1, sensitivity='adjoint', solver='euler')

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

class MD_LSTM(MD_Model):

	def __init__(self, hparams, scaling=None):

		MD_Model.__init__(self, hparams)

		self.lstm = LSTM(input_size=self.hparams.in_features, hidden_size=hparams.num_hidden, num_layers=hparams.num_layers - 1,
				 batch_first=True)
		self.out_emb = Linear(self.hparams.num_hidden, self.hparams.in_features, bias=True)

		if scaling is not None:
			self.y_mu = scaling['y_mu']
			self.y_std = scaling['y_std']
			self.dy_mu = scaling['dy_mu']
			self.dy_std = scaling['dy_std']

	def integration(self, t, x):
		'''
		:param t:
		:param x: [x(0), x(1), x(2)]
		:return:
		'''
		if x.dim() == 2: x = x.unsqueeze(1)
		assert x.dim() == 3

		pred, (h, c) = self.lstm(x)
		dx = self.out_emb(pred)
		out = torch.cat([x, x[:, -1:, :] + dx[:, -1:, :]], dim=1)
		for step in range(t - 1):  # because we add the first entry y0 at the beginning
			pred_t, (h, c) = self.lstm(out[:, -1:], (h, c))
			dx_t = self.out_emb(pred_t)
			out = torch.cat([out, out[:, -1:, :] + dx_t], dim=1)
		return out

class MD_RNN(MD_Model):

	def __init__(self, hparams, scaling=None):

		MD_Model.__init__(self, hparams)

		self.rnn = RNN(input_size=self.hparams.in_features, hidden_size=self.hparams.num_hidden, num_layers=self.hparams.num_layers - 1,
			       batch_first=True)
		self.out_emb = Linear(self.hparams.num_hidden, self.hparams.in_features, bias=True)

		if scaling is not None:
			self.y_mu = scaling['y_mu']
			self.y_std = scaling['y_std']
			self.dy_mu = scaling['dy_mu']
			self.dy_std = scaling['dy_std']

	def integration(self, t, x):
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
		out = torch.cat([x, x[:, -1:, :] + dx[:, -1:, :]], dim=1)
		for step in range(t - 1):  # because we add the first entry y0 at the beginning
			pred_t, h = self.rnn(out[:, -1:], h)
			dx_t = self.out_emb(pred_t)
			out = torch.cat([out, out[:, -1:, :] + dx_t], dim=1)

		'''
		out = [y1' y2' y3' y4' | y5' y6' y7'] -> [y0 y1' y2' y3' y4' | y5' y6' y7']
		'''

		return out

# Bidirectional Models
class MD_BiModel(Module):

	def __init__(self, hparams):

		self.hparams = hparams

		super(MD_BiModel, self).__init__()

	def criterion(self, _pred, _target, mode=None):
		assert _pred.shape == _target.shape, f'{_pred.shape=} {_target.shape=}'
		assert torch.sum(_pred[:, :self.hparams.input_length] - _target[:, :self.hparams.input_length]) == 0, f" y[0] aren't the same"
		assert torch.sum(_pred[:, -self.hparams.input_length:] - _target[:, -self.hparams.input_length:]) == 0, f" y[-1] aren't the same"

		if mode is None:
			# only overwrite mode if its not given
			mode = self.hparams.criterion

		if mode == 'T':
			in_length, out_length = self.hparams.input_length, self.hparams.output_length_train
			total_length = out_length + 2 * in_length
			''' Two entries if total output length is even and one if its odd'''
			T = range(total_length // 2 - 1, total_length // 2 + 1) if total_length % 2 == 0 else total_length // 2
			return F.mse_loss(_pred[:, T], _target[:, T])
		elif mode == 't':
			return F.mse_loss(_pred, _target)
		else:
			raise Exception(f'Wrong Criterion chosen: {self.hparams.criterion}')

	def split_initfinal_conditions(self, x):

		assert x.shape[1] == 2 * self.hparams.input_length
		assert x.dim() == 3

		x1, x2 = torch.chunk(x, chunks=2, dim=1)
		input_length = x1.shape[1]
		BS, Feat = x.shape[0], x.shape[2]

		x_for, x_back = x1, x2.flip([1])  # flip the backward conditioning so that we can simply pass the data forward in time
		assert (x_back[:, -1, :] == x2[:, 0, :]).all()

		return x_for, x_back

	def integration(self, t, x):
		raise NotImplementedError

	def forward_solution(self,t,x):
		'''
		t: number of integration steps
		x: initial condition
		'''
		return self.integration(t=t, x=x)

	def backward_solution(self, t, x):
		'''
		t: number of integration steps
		x: initial condition
		'''
		velocity_flip = torch.ones(x.shape[-1], device=x.device)  # [ 1 1 1 1 ]
		velocity_flip[velocity_flip.shape[-1] // 2:] = -1  # [ 1 1 -1 -1 ]
		x_back = x * velocity_flip
		pred_back = self.integration(t=t, x=x_back).flip([1])
		pred_back = pred_back * velocity_flip

		return pred_back

	def forward(self, t, x):

		x_for, x_back = self.split_initfinal_conditions(x)
		BS, input_length, Feat = x_for.shape

		'''Forwards Solution'''
		pred_for = self.forward_solution(t=input_length + t, x=x_for)

		'''Backwards Solution'''
		pred_back = self.backward_solution(t=input_length + t, x=x_back)

		out = self.adiabatic_convection_interpolation(input_length=input_length, output_length=t, x_for=pred_for, x_back=pred_back)

		assert out.shape == (BS, 2 * input_length + t, Feat)
		return out


	def adiabatic_convection_interpolation(self, input_length, output_length, x_for, x_back):
		'''
		input_length: scalar
		output_length: scalar
		x: input data to forward pass to extract dtype and device
		For input_length=3 and output_length=3 we have forward_weights=[ 1 1 1 | 0.75, 0.5, 0.25 | 0 0 0 ]
		For input_length=3 and output_length=4 we have forward_weights=[ 1 1 1 | 0.8, 0.6, 0.4 0.2 | 0 0 0 ]
		'''
		forward_weights = torch.linspace(start=1, end=0, steps=output_length + 2, dtype=x_for.dtype, device=x_for.device)[1:-1]
		forward_weights = torch.cat([torch.ones(input_length, dtype=x_for.dtype, device=x_for.device),
					     forward_weights,
					     torch.zeros(input_length, dtype=x_for.dtype, device=x_for.device)])
		backward_weights = forward_weights.flip([0])

		''' [T] -> stack -> [T, 2] -> [1,T,1,2] <=> pred.shape=[BS, T, Feat, 2] '''
		weights = torch.stack([forward_weights, backward_weights], dim=-1).unsqueeze(0).unsqueeze(-2)

		pred = torch.stack([x_for, x_back], dim=-1)
		out = torch.sum(pred * weights, dim=-1)

		return out

class MD_BiDirectional_ODE(MD_BiModel):

	def __init__(self, hparams, scaling=None):

		MD_BiModel.__init__(self, hparams)

		if scaling is not None:
			self.y_mu = scaling['y_mu']
			self.y_std = scaling['y_std']
			self.dy_mu = scaling['dy_mu']
			self.dy_std = scaling['dy_std']

		self.md_ode = MD_ODE(hparams=hparams, scaling=scaling)

	def integration(self, t, x):
		return self.md_ode(t, x) # for whatever reason we have to pass on 't+1'

class MD_BiDirectional_RNN(MD_BiModel):

	def __init__(self, hparams, scaling=None):

		MD_BiModel.__init__(self, hparams)

		self.md_rnn = MD_RNN(hparams=hparams, scaling=scaling)

		if scaling is not None:
			self.y_mu = scaling['y_mu']
			self.y_std = scaling['y_std']
			self.dy_mu = scaling['dy_mu']
			self.dy_std = scaling['dy_std']

	def integration(self, t, x):
		'''
		:param t:
		:param x: [x(0), x(1), x(2)]
		:return:
		'''
		return self.md_rnn.forward(t, x)

class MD_BiDirectional_LSTM(MD_BiModel):

	def __init__(self, hparams, scaling=None):

		MD_BiModel.__init__(self, hparams)

		self.md_lstm = MD_LSTM(hparams=hparams, scaling=scaling)

		if scaling is not None:
			self.y_mu = scaling['y_mu']
			self.y_std = scaling['y_std']
			self.dy_mu = scaling['dy_mu']
			self.dy_std = scaling['dy_std']

	def integration(self, t, x):
		'''
		:param t:
		:param x: [x(0), x(1), x(2)]
		:return:
		'''
		return self.md_lstm.forward(t, x)

class MD_BiDirectional_Hamiltonian(MD_BiModel):

	def __init__(self, hparams, scaling=None):
		MD_BiModel.__init__(self, hparams)

		if scaling is not None:
			self.y_mu = scaling['y_mu']
			self.y_std = scaling['y_std']
			self.dy_mu = scaling['dy_mu']
			self.dy_std = scaling['dy_std']

		self.md_hamiltonian = MD_Hamiltonian(hparams=hparams, scaling=scaling)


	def integration(self, t, x):
		return self.md_hamiltonian(t, x)  # for whatever reason we have to pass on 't+1'

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