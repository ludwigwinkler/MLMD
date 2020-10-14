import future, sys, os, datetime, argparse, warnings, inspect
# print(os.path.dirname(sys.executable))
import torch
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

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)
torch.set_default_dtype(torch.double)

sys.path.append("../../..")  # Up to -> KFAC -> Optimization -> PHD

import scipy
import scipy as sp
from scipy.io import loadmat as sp_loadmat
import copy

cwd = os.path.abspath(os.getcwd())
os.chdir(cwd)

from DiffEqNets.MolecularDynamics.src.MD_HyperparameterParser import HParamParser


class SinCos_DataModule:

	def __init__(self):
		self.prepare_data()

	def prepare_data(self, *args, **kwargs):
		t = torch.linspace(0, 10, 50).reshape(1, -1, 1)

		x = torch.cat([t.cos(), t.sin()], dim=-1)
		v = (x[:, 1:] - x[:, :-1])

		self.data = torch.cat([x[:, :-1, :], v], dim=-1)

		assert self.data.dim() == 3
		assert self.data.shape[-1] == 4

	def setup(self):
		pass

class VerletIntegrator:

	def __init__(self, hparams=None):

		# dm = load_dm_data(hparams)
		dm = SinCos_DataModule()

		self.x, _ 	= torch.chunk(dm.data, chunks=2, dim=-1)
		'''
		Forward Diff: v = x(n+1) - x(n) 
		New Index: 0 -> x.length-1
		i.e. x[1] - x[0]
		'''
		self.v 	= (self.x[:,1:]-self.x[:,:-1])
		'''
		Central Difference: a = ( x(n+1) - 2*x(n) + x(n-1) ) / dt**2
		New Index: 1 -> x.length-1
		i.e. x[2] - 2 * x[1] + x[0]
		'''
		self.a = self.x[:,2:,:] - 2*self.x[:,1:-1,:] + self.x[:,:-2,:]

		self.check_integrator()

	def check_integrator(self):
		'''
		self.x/v/a are of shape [#TimeSeries, #TimeSteps, #Features]
		'''
		assert self.x.dim()==3
		assert self.v.dim()==3

		x = self.x.squeeze(0) # drop the TimeSeries dimension
		v = self.v.squeeze(0) # drop the TimeSeries dimension
		a = self.a.squeeze(0) # drop the TimeSeries dimension
		T = x.shape[0]
		euler 	= torch.cat([x[:1], x[0] + torch.cumsum(v, dim=0)], dim=0) # [x[0], x[0] + dx[0], x[0] + dx[0] + dx[1] ... ]
		verlet 	= torch.cat([x[1:2], x[1] + torch.cumsum(v[1:], dim=0) + 0.5*torch.cumsum(a, dim=0)], dim=0)

		print(f"{verlet.shape=}")

		plt.plot(x[:,:3], label='True')
		# plt.scatter(self.x[0,-T:,:3])
		plt.plot(torch.arange(T), euler[:,:3], '--o', label='Euler')
		plt.plot(torch.arange(T-1)+1, verlet[:,:3], '-*', label='Verlet')
		plt.xticks(torch.arange(x.shape[0]))
		plt.yticks(torch.linspace(-1,1, 21))
		plt.legend()
		plt.grid(which='major')
		plt.show()


solver = VerletIntegrator()

