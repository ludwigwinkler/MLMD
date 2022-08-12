import future, sys, os, datetime, argparse, math
# print(os.path.dirname(sys.executable))
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch
from torch.nn import Module, Parameter
from torch.nn import Linear, Tanh, ReLU
import torch.nn.functional as F

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor
Scalar = torch.scalar_tensor

torch.set_printoptions(precision=10, sci_mode=False)
np.set_printoptions(precision=10, suppress=True)

sys.path.append("../../..")  # Up to -> KFAC -> Optimization -> PHD

import scipy
import scipy as sp
from scipy.io import loadmat as sp_loadmat
import copy

class SimMD:

	def __init__(self, dt):

		self.dt = torch.scalar_tensor(dt)
		self.steps = 0

	def step(self, x):
		raise NotImplementedError

	def forecast(self, x, steps):
		raise NotImplementedError

class CosineDynamicalSystem(SimMD):

	def __init__(self, dt, freq):
		super().__init__(dt)

		self.freq = torch.scalar_tensor(freq)
		self.eps = 1e-2

	def get_t_from_posvel(self, pos, vel):
		'''
		Determines position and velocity of cosine dynamics and limits pos and vel to valid range
		@param pos:
		@param vel:
		@return:
		'''

		'''
		x = cos(w*t) -> cos^-1(x) / w = t
		'''
		t = torch.arccos(pos.clamp(-1+self.eps, 1- self.eps))  # [-1, 1] -> [0, np.pi]
		assert 0 <= t <= np.pi, f"{t=}"
		if vel >= 0: t = np.pi + abs(math.pi - t)  # vel>=0 -> t in [π, 2*π]

		T = 600
		# steps = self.steps%T
		# freq = self.freq * (1-0.025*torch.sin(Scalar(self.steps/T*np.pi)))
		freq = self.freq
		# freq = freq.clamp(max=self.freq)
		t /= freq

		return None, None, t
			
	def step(self, x, plot=False):
		'''
		@param x: Torch Tensor of shape [pos_t, vel_t]
		@return: [pos_{t+1}, vel_{t+1}]
		'''
		assert x.shape==(2,), f"{x.shape=}"

		self.t += self.dt

		vel = -self.freq*torch.sin(self.freq * self.t)
		pos = torch.cos(self.freq * self.t)

		# print(f"{self.t=}")

		return Tensor([pos, vel])

	def __call__(self, T, x0=None, t0=0, plot=False):
		'''
		We strictly use t0 to determine the initial condition
		@param t:
		@param x:
		@param t0:
		@param plot:
		@return:
		'''
		if x0 is not None:
			assert x0.shape==(1,2), f"{x0.shape=}"
			assert x0.dim()==2
		assert t0>=0

		self.t = t0
		x0 = Tensor([torch.cos(self.freq * self.t), -self.freq * torch.sin(self.freq * self.t)])

		traj = [x0]
		for _ in range(T-1):
			traj += [self.step(traj[-1])]

		traj = torch.stack(traj, dim=0)
		assert traj.dim()==2, f"{traj.dim()=}"
		assert traj.shape[0]==T, f"{traj.shape[0]=} VS {T=}"

		return traj

class DoubleCosineDynamicalSystem(SimMD):

	def __init__(self, dt, freq):
		super().__init__(dt)

		self.freq1 = torch.scalar_tensor(freq)
		self.freq2 = torch.scalar_tensor(freq*2)
		self.scale2 = torch.scalar_tensor(0.2)
		self.eps = 1e-2

	def get_t_from_posvel(self, pos, vel):
		'''
		Determines position and velocity of cosine dynamics and limits pos and vel to valid range
		@param pos:
		@param vel:
		@return:
		'''

		'''
		x = cos(w*t) -> cos^-1(x) / w = t
		'''
		t = torch.arccos(pos)  # [-1, 1] -> [0, np.pi]
		if vel >= 0: t = np.pi + abs(math.pi - t)  # vel>=0 -> t in [π, 2*π]

		T = 600
		# steps = self.steps%T
		# freq = self.freq * (1-0.025*torch.sin(Scalar(self.steps/T*np.pi)))
		freq = self.freq
		# freq = freq.clamp(max=self.freq)
		t /= freq

		return None, None, t

	def step(self, x, plot=False):
		'''
		@param x: Torch Tensor of shape [pos_t, vel_t]
		@return: [pos_{t+1}, vel_{t+1}]
		'''
		assert x.shape == (2,), f"{x.shape=}"

		self.t += self.dt

		vel = -self.freq1 * torch.sin(self.freq1 * self.t) - self.scale2 * self.freq2 * torch.sin(self.freq2 * self.t)
		pos = torch.cos(self.freq1 * self.t) + self.scale2 * torch.cos(self.freq2 * self.t)

		# print(f"{self.t=}")

		return Tensor([pos, vel])

	def __call__(self, T, x0=None, t0=0, plot=False):
		'''
		We strictly use t0 to determine the initial condition
		@param t:
		@param x:
		@param t0:
		@param plot:
		@return:
		'''
		if x0 is not None:
			assert x0.shape == (1, 2), f"{x0.shape=}"
			assert x0.dim() == 2
		assert t0 >= 0

		self.t = t0
		x0 = Tensor([torch.cos(self.freq1 * self.t) + self.scale2 * torch.cos(self.freq2 * self.t),
			     -self.freq1 * torch.sin(self.freq1 * self.t) - self.scale2 * self.freq2 * torch.sin(self.freq2 * self.t)])

		traj = [x0]
		for _ in range(T - 1):
			traj += [self.step(traj[-1])]

		traj = torch.stack(traj, dim=0)
		assert traj.dim() == 2, f"{traj.dim()=}"
		assert traj.shape[0] == T, f"{traj.shape[0]=} VS {T=}"

		return traj

class ModulatedCosineDynamicalSystem(SimMD):

	def __init__(self, dt, freq):
		super().__init__(dt)

		self.freq1 = torch.scalar_tensor(freq)
		self.freq2 = torch.scalar_tensor(freq*0.5)
		self.eps = 1e-2

	def get_t_from_posvel(self, pos, vel):
		'''
		Determines position and velocity of cosine dynamics and limits pos and vel to valid range
		@param pos:
		@param vel:
		@return:
		'''

		'''
		x = cos(w*t) -> cos^-1(x) / w = t
		'''
		t = torch.arccos(pos)  # [-1, 1] -> [0, np.pi]
		if vel >= 0: t = np.pi + abs(math.pi - t)  # vel>=0 -> t in [π, 2*π]

		T = 600
		# steps = self.steps%T
		# freq = self.freq * (1-0.025*torch.sin(Scalar(self.steps/T*np.pi)))
		freq = self.freq
		# freq = freq.clamp(max=self.freq)
		t /= freq

		return None, None, t

	def step(self, x, plot=False):
		'''
		@param x: Torch Tensor of shape [pos_t, vel_t]
		@return: [pos_{t+1}, vel_{t+1}]
		'''
		assert x.shape == (2,), f"{x.shape=}"

		self.t += self.dt

		vel = -self.freq1 * torch.sin(self.freq1 * self.t) -self.freq2 * torch.sin(self.freq2 * self.t)
		pos = torch.cos(self.freq1 * self.t) + torch.cos(self.freq2 * self.t)

		# print(f"{self.t=}")

		return Tensor([pos, vel])

	def __call__(self, T, x0=None, t0=0, plot=False):
		'''
		We strictly use t0 to determine the initial condition
		@param t:
		@param x:
		@param t0:
		@param plot:
		@return:
		'''
		if x0 is not None:
			assert x0.shape == (1, 2), f"{x0.shape=}"
			assert x0.dim() == 2
		assert t0 >= 0

		self.t = t0
		x0 = Tensor([torch.cos(self.freq1 * self.t) + torch.cos(self.freq2 * self.t),
			     -self.freq1 * torch.sin(self.freq1 * self.t) - -self.freq2 * torch.sin(self.freq2 * self.t)])

		traj = [x0]
		for _ in range(T - 1):
			traj += [self.step(traj[-1])]

		traj = torch.stack(traj, dim=0)
		assert traj.dim() == 2, f"{traj.dim()=}"
		assert traj.shape[0] == T, f"{traj.shape[0]=} VS {T=}"

		return traj


if __name__ == "__main__":
	T = 3*np.pi
	steps = 500
	dyn = CosineDynamicalSystem(dt=T/steps)

	t_0 = torch.scalar_tensor(2*np.pi*0.0001)
	x_0 = torch.Tensor([torch.cos(t_0), -torch.sin(t_0)])
	# print(f"True {t=}")
	# dyn(torch.Tensor([torch.cos(t), -torch.sin(t)]))

	dyn(x_0, steps, plot=True)