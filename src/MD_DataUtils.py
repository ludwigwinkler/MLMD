import numbers
import os, shutil, inspect, copy, warnings
from typing import List, Optional, Tuple, Dict, Union
from os import listdir
from os.path import isfile, join, isdir
import math, numpy as  np
from pathlib import Path

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

import torch
import torch.nn.functional as F
import matplotlib
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset

import sklearn
from sklearn.neighbors._kde import KernelDensity

import pandas as pd
import seaborn as sns
import pandas as pd

fontsize = 80
params = {'font.size': fontsize,
	  'legend.fontsize': fontsize,
	  'xtick.labelsize': fontsize,
	  'ytick.labelsize': fontsize,
	  'axes.labelsize': fontsize,
	  'figure.figsize': (20, 10),
	  'text.usetex': True,
	  'mathtext.fontset': 'stix',
	  'font.family': 'STIXGeneral'
	  }

plt.rcParams.update(params)

import urllib.request
from ase import Atoms, Atom

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split, WeightedRandomSampler
from torch.utils.data import Dataset, ConcatDataset
import torch.nn.functional as F

file_path = os.path.dirname(os.path.abspath(__file__)) + '/MD_DataUtils.py'
cwd = os.path.dirname(os.path.abspath(__file__))

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

Tensor = torch.Tensor
Scalar = torch.scalar_tensor

def to_np(_tensor):
	if _tensor is not None:
		assert isinstance(_tensor, torch.Tensor)
		return _tensor.cpu().squeeze().detach().numpy()
	else:
		return None

def traverse_path(full_path: str, depth: Union[numbers.Number, None] = None):
	# assert type(depth)==int and depth<=0, print(f"{depth=}")

	folders = full_path.split('/')
	depths = ['/'.join(folders[:depth]) for depth in range(1,len(folders))]

	if depth is None or depth==0:
		assert os.path.isdir(full_path)
		return full_path
	elif depth <0:
		assert os.path.isdir(depths[depth])
		return depths[depth]



'''
How to sample from variable length sequences [seq1, seq2, seq3 ...]

Cleanest Way:
	Keep three timeseries in trajectory:
	 	Data: [T, F]
	 	ModelID: [T] i.e. [0,0,0,0,...,1,1,1,1,0,0,0,0] with 0:simmd and 1:mlmd
	 	Length of Segments, i.e. [300, 20, 40, 20, ... ] with provides us with the information to split into relevant segments
	 	data.split(length) => Tuple(*seqs)
	Split 
	Construct Sampler from lengths of individual sequences [ seq1_length, seq2_length, seq3_length ... ], i.e. [ 300, 20, 40, 20 ... ]
	Sampler gives an index proportional to the individual sequence lengths
	Sample starting point randomly from individual sequence
	
	Problem: 
	
Hacky Way:
	Chop up every sequence to a predefined length


'''

class VariableTimeSeries_DataSet(Dataset):

	def __init__(self, seqs, input_length=1, output_length=2, traj_repetition=1):

		assert type(seqs) == list
		assert type(input_length) == int
		assert type(output_length) == int
		assert input_length >= 1
		assert output_length >= 1

		self.input_length = input_length
		self.output_length = output_length

		self.seqs = seqs
		self.traj_repetition = traj_repetition

	def __getitem__(self, idx):

		''''
		DiffEq Interfaces append the solutions to the starting value such that:
		[y0] -> Solver(t) -> [y0 y1 y2 ... yt]

		Python indexing seq[t0:(t0+T)] includes start and excludes end
		seq[t0:(t0+T)] => seq[t0 t1 ... T-1]
		'''

		total_length = self.output_length + self.input_length

		'''
		Many short timeseries
		'''
		idx = idx % len(self.seqs)  # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
		traj = self.seqs[idx]  # selecting trajectory

		assert total_length <= traj.shape[0], f'{total_length=} !<= {traj.shape[0]=}'
		t0 = np.random.choice(traj.shape[0] - total_length) if traj.shape[0] > total_length else 0  # selecting starting time in trajectory


		y0 = traj[t0:(t0 + self.input_length)]  # selecting corresponding startin gpoint
		target = traj[t0:(t0 + total_length)]  # snippet of trajectory

		# assert target.shape[0]==total_length, f"{target.shape=} VS {total_length}"

		return y0, self.output_length, target

	def __len__(self):
		return len(self.seqs) * self.traj_repetition

	def __repr__(self):
		return f"VariableTimeSeries_DataSet: {[data_.shape[0] for data_ in self.seqs]}"


class BiDirectional_VariableTimeSeries_DataSet(Dataset):

	def __init__(self, seqs, input_length=1, output_length=2, traj_repetition=1):
		assert type(seqs) == list
		assert type(input_length) == int
		assert type(output_length) == int
		assert input_length >= 1
		assert output_length >= 1

		self.input_length = input_length
		self.output_length = output_length

		self.seqs = seqs
		self.traj_repetition = traj_repetition

	def __getitem__(self, idx):
		''''
		DiffEq Interfaces append the solutions to the starting value such that:
		[y0] -> Solver(t) -> [y0 y1 y2 ... yt]

		Python indexing seq[t0:(t0+T)] includes start and excludes end
		seq[t0:(t0+T)] => seq[t0 t1 ... T-1]
		'''

		total_length = self.output_length + 2*self.input_length

		'''
		Many short timeseries
		'''
		idx = idx % len(self.seqs)  # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
		traj = self.seqs[idx]  # selecting trajectory

		# if traj.shape[0] <= total_length:
		# 	print('hi')

		assert traj.shape[0] >= total_length , f'{total_length=} !<= {traj.shape[0]=}'
		t0 = np.random.choice(traj.shape[0] - total_length) if traj.shape[0] > total_length else 0  # selecting starting time in trajectory
		t1 = t0 + self.input_length + self.output_length

		y0 = traj[t0:(t0 + self.input_length)]  # selecting corresponding startin gpoint
		y1 = traj[t1:(t1 + self.input_length)]
		target = traj[t0:(t0 + total_length)]  # snippet of trajectory
		y0 = torch.cat([y0, y1], dim=0)

		# assert target.shape[0]==total_length, f"{target.shape=} VS {total_length}"

		return y0, self.output_length, target

	def __len__(self):
		return len(self.seqs) * self.traj_repetition

	def __repr__(self):
		return f"VariableTimeSeries_DataSet: {[data_.shape[0] for data_ in self.seqs]}"


class TimeSeries_DataSet(Dataset):

	def __init__(self, data, input_length=1, output_length=2, output_length_sampling=False, traj_repetition=1, sample_axis=None):

		# assert data.dim() == 3, f'Data.dim()={data.dim()} and not [trajs, steps, features]'
		assert type(input_length) == int
		assert type(output_length) == int
		assert input_length >= 1
		assert output_length >= 1

		if sample_axis is not None:
			assert sample_axis in ['trajs', 'timesteps'], f'Invalid axis ampling'

		self.input_length = input_length
		self.output_length = output_length
		self.output_length_sampling = output_length_sampling
		self.output_length_samplerange = [1, output_length + 1]

		self.data = data
		self.traj_repetition = traj_repetition

		if sample_axis is None:
			if self.data.shape[0] * self.traj_repetition >= self.data.shape[1]:  # more trajs*timesteps than timesteps
				self.sample_axis = 'trajs'
			# print(f'Sampling along trajectory axis {self.data.shape} with dataset multiplier {self.traj_repetition} ->{self.data.shape[0]*self.traj_repetition}')
			elif self.data.shape[0] * self.traj_repetition < self.data.shape[1]:  # more timesteps than trajs
				self.sample_axis = 'timesteps'
			# print(f'Sampling along timestep axis {self.data.shape}->{self.data.shape[1]}')
			else:
				raise ValueError('Sample axis not defined in data set')

		elif sample_axis is not None and sample_axis in ['trajs', 'timesteps']:
			self.sample_axis = sample_axis

	def __getitem__(self, idx):

		''''
		DiffEq Interfaces append the solutions to the starting value such that:
		[y0] -> Solver(t) -> [y0 y1 y2 ... yt]

		Python indexing seq[t0:(t0+T)] includes start and excludes end
		seq[t0:(t0+T)] => seq[t0 t1 ... T-1]
		'''

		if hasattr(self, 'sampled_output_length'):
			output_length = self.sampled_output_length
		else:
			output_length = self.output_length

		total_length = output_length + self.input_length

		if self.sample_axis == 'trajs':
			'''
			Many short timeseries
			'''
			idx = idx % self.data.shape[0]  # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
			traj = self.data[idx]  # selecting trajectory

			assert (traj.shape[0] - total_length) >= 0, f'Trajectory time dimension {traj.shape[0]} is smaller than output length {output_length}'
			t0 = np.random.choice(traj.shape[0] - total_length) if (traj.shape[0] - total_length) > 0 else 0  # selecting starting time in trajectory

		elif self.sample_axis == 'timesteps':
			'''
			Few short timeseries
			'''

			traj_index = np.random.choice(self.data.shape[0])  # Randomly select one of the few timeseries
			traj = self.data[traj_index]  # select the timeseries

			t0 = idx % (self.data.shape[1] - total_length)  # we're sampling from the timesteps

		y0 = traj[t0:(t0 + self.input_length)]  # selecting corresponding startin gpoint
		target = traj[t0:(t0 + total_length)]  # snippet of trajectory

		# assert target.shape[0]==total_length, f"{target.shape=} VS {total_length}"

		return y0, output_length, target

	def __repr__(self):
		return f"TimSeries DataSet: {self.data.shape} Sampling: {self.sample_axis}"

	def __len__(self):

		if self.sample_axis == 'trajs':
			return self.data.shape[0] * self.traj_repetition
		elif self.sample_axis == 'timesteps':
			return (self.data.shape[1] - (self.input_length + self.output_length))
		else:
			raise ValueError('Sample axis not defined in data set not defined')


class BiDirectional_TimeSeries_DataSet(Dataset):

	def __init__(self, data, input_length=1, output_length=2, output_length_sampling=False, traj_repetition=1, sample_axis=None):

		assert data.dim() == 3, f'Data.dim()={data.dim()} and not [trajs, steps, features]'
		assert type(input_length) == int
		assert type(output_length) == int
		assert input_length >= 1
		assert output_length >= 1

		if sample_axis is not None:
			assert sample_axis in ['trajs', 'timesteps'], f'Invalid axis ampling'

		self.input_length = input_length
		self.output_length = output_length
		self.output_length_sampling = output_length_sampling
		self.output_length_samplerange = [1, output_length + 1]

		self.data = data
		self.traj_repetition = traj_repetition

		if sample_axis is None:
			if self.data.shape[0] * self.traj_repetition >= self.data.shape[1]:  # more trajs*timesteps than timesteps
				self.sample_axis = 'trajs'
			# print(f'Sampling along trajectory axis {self.data.shape} with dataset multiplier {self.traj_repetition} ->{self.data.shape[0]*self.traj_repetition}')
			elif self.data.shape[0] * self.traj_repetition < self.data.shape[1]:  # more timesteps than trajs
				self.sample_axis = 'timesteps'
			# print(f'Sampling along timestep axis {self.data.shape}->{self.data.shape[1]}')
			else:
				raise ValueError('Sample axis not defined in data set')

		elif sample_axis is not None and sample_axis in ['trajs', 'timesteps']:
			self.sample_axis = sample_axis

	def sample_output_length(self):

		print(f"{self.sample_output_length=}")

		if self.output_length_sampling:
			self.sampled_output_length = np.random.randint(int(self.output_length_samplerange[0]), int(self.output_length_samplerange[1]))

	def update_output_length_samplerange(self, low=0.1, high=0.5, mode='add'):

		assert mode in ['add', 'set'], 'mode is not set correctly'

		cur_low, cur_high = self.output_length_samplerange[0], self.output_length_samplerange[1]

		if mode == 'add':
			if cur_high + high < self.__len__(): cur_high += high
			if cur_low + low < cur_high: cur_low += low
			self.output_length_samplerange = np.array([cur_low, cur_high])
		elif mode == 'set' and low < high:
			assert high < self.__len__()
			self.output_length_samplerange = np.array([low, high])
		else:
			raise ValueError('Incorrect inputs to update_batchlength_samplerange')

	def __getitem__(self, idx):

		if hasattr(self, 'sampled_output_length'):
			output_length = self.sampled_output_length
		else:
			output_length = self.output_length

		total_length = output_length + 2*self.input_length

		if self.sample_axis == 'trajs':
			'''
			Many short timeseries
			'''
			idx = idx % self.data.shape[0]  # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
			traj = self.data[idx]  # selecting trajectory

			assert (traj.shape[0] - total_length) >= 0, f' trajectory length {traj.shape[0]} is smaller than output_length {output_length}'
			t0 = np.random.choice(traj.shape[0] - total_length)  # selecting starting time in trajectory
			t1 = t0 + self.input_length + self.output_length

		elif self.sample_axis == 'timesteps':
			'''
			Few short timeseries
			'''

			traj_index = np.random.choice(self.data.shape[0])  # Randomly select one of the few timeseries
			traj = self.data[traj_index]  # select the timeseries

			t0 = idx % (self.data.shape[1] - total_length)  # we're sampling from the timesteps
			t1 = t0 + self.input_length + self.output_length
			assert (t0 + total_length) < self.data.shape[1]

		'''
		y0 + input_length | output_length | y1 + input_length
		'''

		y0 = traj[t0:(t0 + self.input_length)]  # selecting corresponding starting point
		y1 = traj[t1:(t1 + self.input_length)]  # selecting corresponding starting point

		target = traj[t0:(t0 + total_length)]  # snippet of trajectory


		assert y0.shape[0] == self.input_length
		assert y1.shape[0] == self.input_length
		assert target.shape[0] == total_length
		assert F.mse_loss(y0, target[:self.input_length]) == 0, f'{F.mse_loss(y0, target[0])=}'
		assert F.mse_loss(y1, target[-self.input_length:]) == 0, f'{F.mse_loss(y1[0], target[-1])=}'

		y0 = torch.cat([y0, y1], dim=0)

		# plt.scatter([y0[0,0].numpy(), y0[1,0].numpy()], [y0[0,1].numpy(), y0[1,1].numpy()])
		# plt.plot(target[:,0], target[:,1])
		# plt.show()
		# exit()

		return y0, output_length, target

	def __len__(self):

		# assert self.data.dim()==3

		if self.sample_axis == 'trajs':
			return self.data.shape[0] * self.traj_repetition
		elif self.sample_axis == 'timesteps':
			return (self.data.shape[1] - (self.input_length + self.output_length + self.input_length))
		else:
			raise ValueError('Sample axis not defined in data set not defined')


class Sequential_BiDirectional_TimeSeries_DataSet(Dataset):

	def __init__(self, data, input_length=1, output_length=2, sample_axis=None):
		assert data.dim() == 2, f'Data.dim()={data.dim()} and not [steps, features]'
		assert type(input_length) == int
		assert type(output_length) == int
		assert input_length >= 1
		assert output_length >= 1

		if sample_axis is not None:
			assert sample_axis in ['trajs', 'timesteps'], f'Invalid axis ampling'

		self.input_length = input_length
		self.output_length = output_length

		T = data.shape[0]
		last_part = T % (input_length+output_length)
		data = data[:(T-last_part)]
		assert data.dim()==2, f'{data.shape=}'

		data_ = torch.stack(data.chunk(chunks=T // (input_length + output_length), dim=0)[:-1])  # dropping the last, possibly wrong length time series sample
		data_ = torch.cat([data_[:-1], data_[1:, :input_length]], dim=1)

		assert data_.shape[1] == (2*input_length + output_length), f"{data_.shape=}"
		# plt.plot(data_[:3,:(-input_length),:3].flatten(0,1))
		# plt.show()
		assert F.mse_loss(data_[:3, :(-input_length)].flatten(0, 1), data[:(3*(input_length+output_length))])==0, f'Data was not properly processed into segments'
		# exit()

		self.data = data_

	def __getitem__(self, idx):
		'''
		Many short timeseries
		'''
		idx = idx % self.data.shape[0]  # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
		traj = self.data[idx]  # selecting trajectory

		'''
		y0 + input_length | output_length | y1 + input_length
		'''

		y0 = traj[:self.input_length]  # selecting corresponding starting point
		y1 = traj[-self.input_length:]  # selecting corresponding starting point

		target = traj  # snippet of trajectory
		assert F.mse_loss(y0, target[:self.input_length]) == 0, f'{F.mse_loss(y0, target[0])=}'
		assert F.mse_loss(y1, target[-self.input_length:]) == 0, f'{F.mse_loss(y1[0], target[-1])=}'

		y0 = torch.cat([y0, y1], dim=0)

		return y0, self.output_length, target

	def __len__(self):
		return self.data.shape[0]


class Sequential_TimeSeries_DataSet(Dataset):

	def __init__(self, data, input_length=1, output_length=2, sample_axis=None):

		assert data.dim() == 2, f'Data.dim()={data.dim()} and not [steps, features]'
		assert type(input_length) == int
		assert type(output_length) == int
		assert input_length >= 1
		assert output_length >= 1

		if sample_axis is not None:
			assert sample_axis in ['trajs', 'timesteps'], f'Invalid axis ampling'

		self.input_length = input_length
		self.output_length = output_length

		# print(f"Sequential TimeSeriesData: {data.shape=}")

		T = data.shape[0]
		last_part = T % (input_length + output_length)
		data = data[:(T - last_part)]
		assert data.dim() == 2, f'{data.shape=}'
		data_ = torch.stack(data.chunk(chunks=T // (input_length + output_length), dim=0))  # dropping the last, possibly wrong length time series sample


		assert data_.shape[1] == (input_length + output_length), f"{data_.shape=}"
		# plt.plot(data_[:3,:(-input_length),:3].flatten(0,1))
		# plt.show()
		assert F.mse_loss(data_[:3].flatten(0, 1),
				  data[:(3 * (input_length + output_length))]) == 0, f'Data was not properly processed into segments'
		# exit()

		self.data = data_

	def __getitem__(self, idx):
		'''
		Many short timeseries
		'''
		idx = idx % self.data.shape[0]  # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
		traj = self.data[idx]  # selecting trajectory

		'''
		y0 + input_length | output_length
		'''

		y0 = traj[:self.input_length]  # selecting corresponding starting point

		target = traj  # snippet of trajectory
		assert F.mse_loss(y0, target[:self.input_length]) == 0, f'{F.mse_loss(y0, target[0])=}'
		# if self.input_length == 1: y0.squeeze_(0)

		return y0, self.output_length, target

	def __len__(self):
		return self.data.shape[0]


class MD_DataSet(LightningDataModule):

	def __init__(self, config):

		'''
		LightningDataModule reserves the attribute hparams for itself and it cannot be set
		'''

		super().__init__()

		self.data_path = Path(__file__).absolute().parent.parent / ('data/' + config.dataset)
		self.data_str = config.dataset

		self.config = config

		self.sequential_sampling=False

	def load_and_process_data(self):

		raise NotImplementedError()

	def normalize(self, x):
		if hasattr(self, 'data_mean') and hasattr(self, 'data_std'):
			return (x - self.data_mean)/self.data_std
		else:
			raise Warning('Normalization unsuccessfull: data_mean and data_std not existant')
			return x

	def unnormalize(self, x):
		if hasattr(self, 'data_mean') and hasattr(self, 'data_std'):
			return x * self.data_std.to(x.device) + self.data_mean.to(x.device)
		else:
			raise Warning('Normalization unsuccessfull: data_mean and data_std not existant')
			return x

	def setup(self, stage: Optional[str] = None):

		self.load_and_process_data()

		assert self.data.dim() == 3
		assert self.data.shape[0] == 1

		'''
		data.shape = [ts, timesteps, features]
		'''
		self.data_mean = self.data.mean(dim=[0,1])
		self.data_std = self.data.std(dim=[0,1]) + 1e-8

		self.data_norm = self.normalize(self.data)

		'''
		val_split: point in timeseries from where we consider it the validation trajectory
		'''
		val_split = int(self.data_norm.shape[1] * self.config.val_split)
		data_train = self.data_norm[:, :int(val_split*self.config.pct_dataset)] # future reduce val_split by training percentage
		data_val = self.data_norm[:, val_split:]

		self.y_mu 	= data_train.data.mean(dim=[0, 1])
		self.y_std 	= data_train.data.std(dim=[0, 1])

		self.dy 	= (data_train.data[:, 2:, :] - data_train.data[:, :-2, :]) / 2
		self.dy_mu 	= self.dy.mean(dim=[0, 1]).unsqueeze(0)  # shape = [bs, f]
		self.dy_std 	= self.dy.std(dim=[0, 1]).unsqueeze(0)  # shape = [bs, f]

		if 'bi' in self.config.model:
			self.data_train = BiDirectional_TimeSeries_DataSet(data=data_train,
									   input_length=self.config.input_length,
									   output_length=self.config.output_length_train,
									   output_length_sampling=self.config.output_length_sampling,
									   traj_repetition=self.config.train_traj_repetition,
									   sample_axis='timesteps')
			self.data_val = BiDirectional_TimeSeries_DataSet(data=data_val,
									 input_length=self.config.input_length,
									 output_length=self.config.output_length_val,
									 output_length_sampling=self.config.output_length_sampling,
									 traj_repetition=1,
									 sample_axis='timesteps')

		else:  # the unidirectional case
			self.data_train = TimeSeries_DataSet(data=data_train,
							     input_length=self.config.input_length,
							     output_length=self.config.output_length_train,
							     output_length_sampling=self.config.output_length_sampling,
							     traj_repetition=self.config.train_traj_repetition,
							     sample_axis='timesteps')
			self.data_val = TimeSeries_DataSet(data=data_val,
							   input_length=self.config.input_length,
							   output_length=self.config.output_length_val,
							   output_length_sampling=self.config.output_length_sampling,
							   traj_repetition=1,
							   sample_axis='timesteps')

	def train_dataloader(self, *args, **kwargs) -> DataLoader:

		dataloader = DataLoader(self.data_train, batch_size=self.config.batch_size, shuffle=True, num_workers=self.config.num_workers)

		return dataloader

	def val_dataloader(self, *args, **kwargs) -> DataLoader:
		return DataLoader(self.data_val, batch_size=self.config.batch_size*2, num_workers=self.config.num_workers)

	def plot_prediction(self, y: list, pred: list, y0=None, t0=None, num_sequential_samples=None, extra_title='', extra_text=''):
		'''
		pred: the prediction by the neural network
		'''

		assert isinstance(y, torch.Tensor)
		assert isinstance(pred, torch.Tensor)
		# assert isinstance(y0, torch.Tensor)
		# assert isinstance(pred, list)
		if y0 is not None: assert isinstance(y0, torch.Tensor), f"{type(y0)=}"

		assert y.dim()==pred.dim()==2

		colors = ['r', 'g', 'b']

		fig, axs = plt.subplots(2,1, sharex=True, figsize=(30,10))
		axs = axs.flatten()

		num_plots = y.shape[-1]//2 # determine how many pos and vel plots there are
		if y0 is not None and t0 is not None: t_y0 = t0[:y0.shape[0]]

		for plot_i in [0, 1, 2][:num_plots]:  # plot the first three at most which are the first three position timeseries
			axs[0].plot(pred[:, plot_i], ls='--', color=colors[plot_i], label='Data' if plot_i == 0 else None)
			axs[0].plot(y[:, plot_i], ls='-', color=colors[plot_i], label='Prediction' if plot_i == 0 else None)
			if y0 is not None and t0 is not None: axs[0].scatter(t_y0, y0[:t_y0.shape[0], plot_i], color=colors[plot_i], s=150, marker='|', label='Initial and Final Conditions' if plot_i == 0 else None)

		axs[0].set_xlabel('Steps')
		axs[0].set_ylabel('$q(t)$')
		axs[0].grid()
		axs[0].set_ylim(-3 if min([pred_.min() for pred_ in pred])<-10 else None, 10 if min([pred_.max() for pred_ in pred]) > 10 else None)

		momenta_index = y.shape[-1] // 2
		for plot_i in [momenta_index, momenta_index+1, momenta_index+2][:num_plots]: # plot the last three at most which are the last three velocity vectors

			axs[1].plot(pred[:, plot_i], ls='--', color=colors[plot_i-momenta_index], label='Data' if plot_i == momenta_index else None)
			axs[1].plot(y[:, plot_i], ls='-', color=colors[plot_i - momenta_index], label='Prediction' if plot_i == momenta_index else None)
			if y0 is not None and t0 is not None: axs[1].scatter(t_y0, y0[:t_y0.shape[0], plot_i], color=colors[plot_i - momenta_index], s=150, marker='|', label='Initial and Final Conditions' if plot_i == momenta_index else None)

		axs[1].set_xlabel('Steps')
		axs[1].set_ylabel('$p(t)$')
		axs[1].grid()
		axs[0].set_ylim(-3 if pred.min() < -10 else None, 10 if pred.max() > 10 else None)

		fig.suptitle(self.config.dataset_nicestr+extra_title)
		plt.legend()

		if extra_text is not None:
			bbox = dict(facecolor='white')
			axs[0].text(1.01, 0.98, extra_text, transform=axs[0].transAxes, va='top', bbox=bbox)
		plt.tight_layout()

		plt.show()

		return fig

	def plot_sequential_prediction(self, y: list, pred: list, y0=None, t0=None, num_sequential_samples=None, extra_title='', extra_text=''):
		'''
		pred: the prediction by the neural network
		'''

		assert isinstance(y, list)
		# assert isinstance(y0, torch.Tensor)
		assert isinstance(pred, list)
		if y0 is not None: assert isinstance(y0, torch.Tensor), f"{type(y0)=}"

		assert y[0].dim()==pred[0].dim()==2

		colors = ['r', 'g', 'b']

		fig, axs = plt.subplots(2,1, sharex=True, figsize=(30,10))
		axs = axs.flatten()

		num_plots = y[0].shape[-1]//2 # determine how many pos and vel plots there are
		if y0 is not None and t0 is not None: t_y0 = t0[:y0.shape[0]]

		for plot_i in [0, 1, 2][:num_plots]:  # plot the first three at most which are the first three position timeseries
			t0_ = 0
			for y_i, pred_i in zip(y, pred):
				t_i = t0_ + torch.arange(y_i.shape[0])
				axs[0].plot(t_i, pred_i[:, plot_i], ls='--', color=colors[plot_i], label='Data' if plot_i == 0 else None)
				if y0 is not None and t0 is not None: axs[0].scatter(t_y0, y0[:t_y0.shape[0], plot_i], color=colors[plot_i], s=150, marker='|', label='Initial and Final Conditions' if plot_i == 0 else None)
				t0_ += y_i.shape[0]

				if num_sequential_samples is not None:
					if t0_ > num_sequential_samples: break


			axs[0].plot(torch.cat(y)[:t0_, plot_i], ls='-', color=colors[plot_i], label='Prediction' if plot_i == 0 else None)

		axs[0].set_xlabel('Steps')
		axs[0].set_ylabel('$q(t)$')
		axs[0].grid()
		axs[0].set_ylim(-3 if min([pred_.min() for pred_ in pred])<-10 else None, 10 if min([pred_.max() for pred_ in pred]) > 10 else None)

		momenta_index = y[0].shape[-1] // 2
		for plot_i in [momenta_index, momenta_index+1, momenta_index+2][:num_plots]: # plot the last three at most which are the last three velocity vectors
			t0_ = 0
			for i, (y_i, pred_i) in enumerate(zip(y, pred)):
				t_i = t0_ + torch.arange(y_i.shape[0])
				axs[1].plot(t_i, pred_i[:, plot_i], ls='--', color=colors[plot_i-momenta_index], label='Data' if plot_i == momenta_index and i==0 else None)
				if y0 is not None and t0 is not None: axs[1].scatter(t_y0, y0[:t_y0.shape[0], plot_i], color=colors[plot_i - momenta_index], s=150, marker='|', label='Initial and Final Conditions' if plot_i == momenta_index and i == 0 else None)
				t0_ += y_i.shape[0]

				if num_sequential_samples is not None:
					if t0_ > num_sequential_samples: break

			axs[1].plot(torch.cat(y)[:t0_, plot_i], ls='-', color=colors[plot_i - momenta_index], label='Prediction' if plot_i == momenta_index and i == 0 else None)


		axs[1].set_xlabel('Steps')
		axs[1].set_ylabel('$p(t)$')
		axs[1].grid()
		axs[0].set_ylim(-3 if min([pred_.min() for pred_ in pred]) < -10 else None, 10 if min([pred_.max() for pred_ in pred]) > 10 else None)

		fig.suptitle(self.config.dataset_nicestr+extra_title)
		plt.legend()

		if extra_text is not None:
			bbox = dict(facecolor='white')
			axs[0].text(1.01, 0.98, extra_text, transform=axs[0].transAxes, va='top', bbox=bbox)
		plt.tight_layout()

		plt.show()

		return fig

	def plot_interatomic_distances_histogram(self, y, pred, str=''):
		'''

		'''
		assert y.dim()==pred.dim()==1, f"{y.shape=} and {pred.shape=}"
		if isinstance(y, torch.Tensor): y = y.numpy()
		if isinstance(pred, torch.Tensor): pred = pred.numpy()
		
		ratio = (12, 18)
		fig, axs = plt.subplots(nrows=2, ncols=1, figsize=ratio, gridspec_kw={'height_ratios': [2, 1]})
		
		# fig = plt.figure(figsize=ratio)
		# fig, ax = plt.subplots(1, 2, figsize=(7, 7))
		# plt.hist(y, density=True, bins=100, alpha=0.5, label='Data')
		# plt.hist(pred, density=True, bins=100, alpha=0.5, label='Interpolation')
		# hist = sns.histplot({'MD': y, 'MLMD': pred}, stat='density', bins=40, legend=True, ax=axs[0])


		y_kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(y.reshape(-1,1))
		y_index = np.linspace(start=y.min(), stop=y.max(), num=200).reshape(-1,1)
		y_kde_estimation = np.exp(y_kde.score_samples(y_index))

		pred_kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(pred.reshape(-1,1))
		pred_index = np.linspace(start=pred.min(), stop=pred.max(), num=200).reshape(-1,1)
		pred_kde_estimation = np.exp(pred_kde.score_samples(pred_index))

		axs[0].plot(np.linspace(y.min(), y.max(), num=200), y_kde_estimation, label='MD', color='blue')
		axs[0].plot(np.linspace(y.min(), y.max(), num=200), pred_kde_estimation, label='MLMD', color='orange')
		diff = y_kde_estimation - pred_kde_estimation
		max_diff = max(abs(diff))*1.2
		axs[1].plot(np.linspace(y.min(), y.max(), num=200), y_kde_estimation-pred_kde_estimation, color='orange')
		axs[1].axhline(y=0, lw=1, color='black')

		axs[0].set_xlabel('$d[\AA]$')
		axs[0].set_ylabel(r'Radial Distribution function (a.u.)')

		axs[1].set_xlabel('$d[\AA]$')
		axs[1].set_ylabel(r'Difference')

		axs[0].spines['right'].set_visible(False)
		axs[0].spines['top'].set_visible(False)
		axs[0].xaxis.set_ticks_position('bottom')
		axs[0].yaxis.set_ticks_position('left')
		axs[1].spines['top'].set_visible(False)
		axs[1].spines['right'].set_visible(False)
		axs[1].xaxis.set_ticks_position('bottom')
		axs[1].yaxis.set_ticks_position('left')
		axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1), useMathText=True)
		# axs[0].set(frame_on=False)
		# axs[1].set(frame_on=False)
		axs[0].legend(frameon=False, handlelength=0.5, handletextpad=0.25)
		plt.tight_layout()
		axs[1].set_ylim(-max_diff, max_diff)

		plt.savefig(f'../visualization/plots/distances_{str}.pdf', format='pdf', bbox_inches='tight')
		plt.savefig(f'../visualization/plots/distances_{str}.svg', format='svg', bbox_inches='tight')
		plt.show()

	def plot_speed_histogram(self, y, pred, str=''):
		'''

		'''

		assert y.dim()==pred.dim()==1, f"{y.shape=} and {pred.shape=}"
		# Catching possible passing of tensor which causes problems with matplotlib later on
		if isinstance(y, torch.Tensor): y = y.numpy()
		if isinstance(pred, torch.Tensor): pred = pred.numpy()

		zoom_in_plot = False
		if zoom_in_plot:

			ratio = (7, 14)
			fig, axs = plt.subplots(2, 1, figsize=ratio)
			bins = 50
			_ = axs[0].hist(y, bins=bins, density=True, alpha=0.66, label='MD', edgecolor='black')
			_ = axs[0].hist(pred, bins=bins, density=True, alpha=0.66, label='MLMD', edgecolor='black')
			axs[0].grid()
			axs[0].set_xlabel(r'$|\vec{\dot{r}}|$')
			axs[0].set_ylabel(r'Probability')
			axs[0].set_xlim(0, 8)
			axs[0].add_patch(matplotlib.patches.Rectangle((0.9, 0.18), 3.2, 0.25, facecolor='none', edgecolor='black'))
			axs[0].legend()

			hist_y = np.histogram(y, bins=bins, normed=True)
			hist_pred = np.histogram(pred, bins=bins, normed=True)
			diff = hist_y[0] - hist_pred[0]
			max_diff = round(max(abs(diff))*1.2,2) # get max, stretch it a bit and round to two decimals

			_ = axs[1].bar(x=hist_y[1][:-1], height=diff, edgecolor='black', width=axs[0].containers[0][0]._width,
						   alpha=0.66)

			axs[1].set_xlabel(r'$|\vec{\dot{r}}|$')
			axs[1].set_ylabel(r'Difference in Probability')
			axs[1].set_xlim(1,4.1)
			axs[1].set_ylim(-max_diff, max_diff)
			axs[1].set_yscale('symlog', base=10, linthresh=0.06)
			axs[1].set_yticks([0.05, 0.025, 0, -0.025, -0.05])
			# axs[1].yaxis.tick_right()
			axs[1].grid()

			fig.suptitle(str)
			plt.tight_layout()

			xy0 = (0.9, 0.18)
			xy1 = (1, max_diff)
			con = matplotlib.patches.ConnectionPatch(axesA=axs[0], axesB=axs[1], xyA=xy0, xyB=xy1,
													 coordsA="data", coordsB="data")
			axs[1].add_artist(con)

			xy0 = (4.1, 0.18)
			xy1 = (4.1, max_diff)
			con = matplotlib.patches.ConnectionPatch(axesA=axs[0], axesB=axs[1], xyA=xy0, xyB=xy1,
													 coordsA="data", coordsB="data")
			axs[1].add_artist(con)

		else:
			
			ratio = (12, 18)
			fig, axs = plt.subplots(2, 1, figsize=ratio, gridspec_kw={'height_ratios': [2, 1]})

			y_kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(y.reshape(-1, 1))
			y_index = np.linspace(start=y.min(), stop=y.max(), num=200).reshape(-1, 1)
			y_kde_estimation = np.exp(y_kde.score_samples(y_index))

			pred_kde = KernelDensity(kernel='gaussian', bandwidth=0.1).fit(pred.reshape(-1, 1))
			pred_index = np.linspace(start=pred.min(), stop=pred.max(), num=200).reshape(-1, 1)
			pred_kde_estimation = np.exp(pred_kde.score_samples(pred_index))

			axs[0].plot(np.linspace(y.min(), y.max(), num=200), y_kde_estimation,  		label='MD', color='blue')
			axs[0].plot(np.linspace(y.min(), y.max(), num=200), pred_kde_estimation, 	label='MLMD', color='orange')
			diff = y_kde_estimation - pred_kde_estimation
			max_diff = max(abs(diff)) * 1.2
			axs[1].plot(np.linspace(y.min(), y.max(), num=200), y_kde_estimation - pred_kde_estimation,  color='orange')
			axs[1].axhline(y=0, lw=1, color='black')
			axs[0].set_xlabel(r'$|\vec{\dot{r}}|$')
			axs[0].set_ylabel(r'Probability')

			axs[0].legend()

			axs[1].set_xlabel(r'$|\vec{\dot{r}}|$')
			axs[1].set_ylabel(r'Difference in Probability')
			axs[0].spines['right'].set_visible(False)
			axs[0].spines['top'].set_visible(False)
			axs[0].xaxis.set_ticks_position('bottom')
			axs[0].yaxis.set_ticks_position('left')
			axs[1].spines['top'].set_visible(False)
			axs[1].spines['right'].set_visible(False)
			axs[1].xaxis.set_ticks_position('bottom')
			axs[1].yaxis.set_ticks_position('left')
			axs[1].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1), useMathText=True)
			# axs[0].set(frame_on=False)
			# axs[1].set(frame_on=False)
			axs[0].legend(frameon=False, handlelength=0.5, handletextpad=0.25)
			axs[1].set_ylim(-max_diff, max_diff)
			plt.tight_layout()

		plt.savefig(f'../visualization/plots/velocities_{str}.pdf', format='pdf', bbox_inches='tight')
		plt.savefig(f'../visualization/plots/velocities_{str}.svg', format='svg', bbox_inches='tight')
		plt.show()

	def plot_spectrum(self, y, pred, extra_title='', extra_text=''):

		assert type(y) == type(pred) == torch.Tensor
		assert y.dim() == pred.dim() == 2

		if self.config.dataset == 'cosine':

			y_power_spectrum_q = (torch.fft.rfft(y[:, 0])).abs().pow(0.5)
			y_power_spectrum_p = (torch.fft.rfft(y[:, 1])).abs().pow(0.5)

			pred_power_spectrum_q = (torch.fft.rfft(pred[:, 0])).abs().pow(0.5)
			pred_power_spectrum_p = (torch.fft.rfft(pred[:, 1])).abs().pow(0.5)

			freqs = torch.linspace(0, 0.5 / self.config.dt * 2 * np.pi, y_power_spectrum_q.shape[0])

			colors = ['r', 'g', 'b']

			fig, axs = plt.subplots(2, 1, sharex=True, figsize=(30, 10))
			axs = axs.flatten()

			num_plots = pred.shape[-1] // 2  # determine how many pos and vel plots there are

			freq_cutoff = 200

			for plot_i in [0, 1, 2][:num_plots]:  # plot the first three at most which are the first three position timeseries

				if plot_i == 0:
					axs[0].plot(freqs[:freq_cutoff], y_power_spectrum_q[:freq_cutoff], ls='-', color=colors[plot_i], label='Data')
					axs[0].plot(freqs[:freq_cutoff], pred_power_spectrum_q[:freq_cutoff], ls='--', color=colors[plot_i], label='Prediction')
				else:
					axs[0].plot(freqs[:freq_cutoff], y_power_spectrum_q[:freq_cutoff], ls='-', color=colors[plot_i])
					axs[0].plot(freqs[:freq_cutoff], pred_power_spectrum_q[:freq_cutoff], ls='--', color=colors[plot_i])

				axs[0].set_xlabel('Frequency')
				axs[0].set_ylabel('Amplitude: $q(t)$')
				axs[0].grid()

				if plot_i == 0:
					axs[1].plot(freqs[:freq_cutoff], y_power_spectrum_p[:freq_cutoff], ls='-', color=colors[plot_i], label='Data')
					axs[1].plot(freqs[:freq_cutoff], pred_power_spectrum_p[:freq_cutoff], ls='--', color=colors[plot_i], label='Prediction')
				else:
					axs[1].plot(freqs[:freq_cutoff], y_power_spectrum_p[:freq_cutoff], ls='-', color=colors[plot_i])
					axs[1].plot(freqs[:freq_cutoff], pred_power_spectrum_p[:freq_cutoff], ls='--', color=colors[plot_i])

				axs[1].set_xlabel('Frequency')
				axs[1].set_ylabel('Amplitude: $p(t)$')
				axs[1].grid()
				axs[1].set_ylim(-3 if pred.min() < -10 else None, 10 if pred.max() > 10 else None)

				fig.suptitle(self.config.dataset_nicestr + extra_title if extra_title!='' else self.config.dataset_nicestr)
				plt.legend()

				if extra_text != '':
					bbox = dict(facecolor='white')
					axs[0].text(1.01, 0.98, extra_text, transform=axs[0].transAxes, va='top', bbox=bbox)
				plt.tight_layout()

			return fig

	def plot_vibrational_spectra(self, y, pred, extra_title='', extra_text='', show=False, save=False):
		'''
		From http://quantum-machine.org/gdml/doc/applications.html#vibrational-spectra
		:param y:
		:param pred:
		:param extra_title:
		:param extra_text:
		:return:
		'''

		import numpy as np
		from scipy.fftpack import fft, fftfreq
		from ase.io.trajectory import Trajectory

		def pdos(V, dt):
			"""
			Calculate the phonon density of states from a trajectory of
			velocities (power spectrum of the velocity auto-correlation
			function).

			Parameters
			----------
			V : :obj:`numpy.ndarray`
			    (dims N x T) velocities of N degrees of freedom for
			    trajetory of length T
			dt : float
			    time between steps in trajectory (fs)

			Returns
			-------
			freq : :obj:`numpy.ndarray`
			    (dims T) Frequencies (cm^-1)
			pdos : :obj:`numpy.ndarray`
			    (dims T) Density of states (a.u.)
			"""

			n_steps = V.shape[1]

			# mean velocity auto-correlation for all degrees of freedom
			vac2 = [np.correlate(v, v, 'full') for v in V]
			vac2 /= np.linalg.norm(vac2, axis=1)[:, None]
			vac2 = np.mean(vac2, axis=0)

			# power spectrum (phonon density of states)
			pdos = np.abs(fft(vac2)) ** 2
			pdos /= np.linalg.norm(pdos) / 2  # spectrum is symmetric

			freq = fftfreq(2 * n_steps - 1, dt) * 33356.4095198152  # Frequency in cm^-1

			return freq[:n_steps], pdos[:n_steps]

		assert y.shape==pred.shape, f"{y.shape=} VS {pred.shape=}"
		assert y.dim()==2, f"{y.dim()=}"

		true_vel = torch.chunk(y, chunks=2, dim=-1)[1]
		pred_vel = torch.chunk(pred, chunks=2, dim=-1)[1]
		true_vel = true_vel.T
		pred_vel = pred_vel.T
		dt = 0.5

		true_freq, true_pdos = pdos(true_vel, dt)
		pred_freq, pred_pdos = pdos(pred_vel, dt)
		# smoothing of the spectrum
		# (removes numerical artifacts due to finite time trunction of the FFT)
		from scipy.ndimage.filters import gaussian_filter1d as gaussian
		true_pdos = gaussian(true_pdos, sigma=25)
		pred_pdos = gaussian(pred_pdos, sigma=25)

		if show:
			fig = plt.figure()

			plt.plot(true_freq, true_pdos, label='Ground Truth')
			plt.plot(pred_freq, pred_pdos, label='Interpolation')
			plt.yticks([])
			plt.xlim(0, 4000)

			plt.box(on=None)
			plt.legend()
			plt.xlabel('Frequency [cm$^{-1}$]')
			plt.ylabel('Density of states [a.u.]')
			plt.title(f'Velocity auto-correlation function\n {extra_title}')
			# plt.savefig('vaf.png')
			plt.show()

			if save:
				plt.savefig(f'vibrationalspectra_interpolation_T{self.hparams.output_length_val}.svg', format='svg')

		return {'frequency': true_freq,'pred_pdos': pred_pdos, 'true_pdos': true_pdos, 'frequency_diff': torch.from_numpy(true_pdos - pred_pdos)}

	def save_as_npz(self, pred, true, conditions, name, path):
		raise NotImplementedError

	def __repr__(self):

		return f'{self.data_str}: [Num Trajectory, Time Steps, Features ]={self.data.shape} features'


class MLMD_Trajectory_DataSet(MD_DataSet):

	def __init__(self, config, data, sample_axis=None, traj_repetition=1):

		MD_DataSet.__init__(self, config)

		assert data.dim()==3
		self.data = data
		self.traj_repetition = traj_repetition

		if sample_axis is None:
			if self.data.shape[0] * self.traj_repetition >= self.data.shape[1]:  # more trajs*timesteps than timesteps
				self.sample_axis = 'trajs'
			# print(f'Sampling along trajectory axis {self.data.shape} with dataset multiplier {self.traj_repetition} ->{self.data.shape[0]*self.traj_repetition}')
			elif self.data.shape[0] * self.traj_repetition < self.data.shape[1]:  # more timesteps than trajs
				self.sample_axis = 'timesteps'
			# print(f'Sampling along timestep axis {self.data.shape}->{self.data.shape[1]}')
			else:
				raise ValueError('Sample axis not defined in data set')

		elif sample_axis is not None and sample_axis in ['trajs', 'timesteps']:
			self.sample_axis = sample_axis

	def load_and_process_data(self):

		if type(self.data) is not torch.Tensor: self.data = torch.from_numpy(self.data)

		if self.data.dim()==2: self.data.unsqueeze_(0)

		assert self.data.dim()==3, f"{self.data.dim()=}"

		self.raw_data = self.data.clone()

	def setup(self, stage: Optional[str] = None):

		self.load_and_process_data()

		assert self.data.dim() == 3

		'''
		val_split: point in timeseries from where we consider it the validation trajectory
		'''
		train_data_size = int(self.data.shape[1] * self.config.val_split * self.config.pct_dataset)
		val_data_size = int(self.data.shape[1] * (1 - self.config.val_split))

		if self.sample_axis=='timesteps':
			val_split = int(self.data.shape[1] * self.config.val_split)
			data_train = self.data[:, :int(val_split * self.config.pct_dataset)]  # future reduce val_split by training percentage
			data_val = self.data[:, int(val_split * self.config.pct_dataset):int(val_split * self.config.pct_dataset + val_data_size)]
		elif self.sample_axis=='trajs':
			val_split = int(self.data.shape[0] * self.config.val_split)
			data_train = self.data[:int(val_split * self.config.pct_dataset)]  # future reduce val_split by training percentage
			data_val = self.data[int(val_split * self.config.pct_dataset):int(val_split * self.config.pct_dataset + val_data_size)]
		'''
		data.shape = [ts, timesteps, features]
		'''
		self.data_mean = self.data.mean(dim=[0, 1])
		self.data_std = self.data.std(dim=[0, 1])

		self.data_norm = (self.data - self.data_mean) / (self.data_std + 1e-8)

		self.y_mu = data_train.data.mean(dim=[0, 1])
		self.y_std = data_train.data.std(dim=[0, 1])

		self.dy = (data_train.data[:, 2:, :] - data_train.data[:, :-2, :]) / 2
		self.dy_mu = self.dy.mean(dim=[0, 1]).unsqueeze(0)  # shape = [bs, f]
		self.dy_std = self.dy.std(dim=[0, 1]).unsqueeze(0)  # shape = [bs, f]

		if 'bi' in self.config.model:
			self.data_train = BiDirectional_TimeSeries_DataSet(data=data_train,
									   input_length=self.config.input_length,
									   output_length=self.config.output_length_train,
									   output_length_sampling=self.config.output_length_sampling,
									   traj_repetition=self.config.train_traj_repetition,
									   sample_axis='trajs')
			self.data_val = BiDirectional_TimeSeries_DataSet(data=data_val,
									 input_length=self.config.input_length,
									 output_length=self.config.output_length_val,
									 output_length_sampling=self.config.output_length_sampling,
									 traj_repetition=1,
									 sample_axis='trajs')

		else:  # the unidirectional case
			self.data_train = TimeSeries_DataSet(data=data_train,
							     input_length=self.config.input_length,
							     output_length=self.config.output_length_train,
							     output_length_sampling=self.config.output_length_sampling,
							     traj_repetition=self.config.train_traj_repetition,
							     sample_axis='trajs')
			self.data_val = TimeSeries_DataSet(data=data_val,
							   input_length=self.config.input_length,
							   output_length=self.config.output_length_val,
							   output_length_sampling=self.config.output_length_sampling,
							   traj_repetition=1,
							   sample_axis='trajs')

	def plot_traj(self):

		pos, vel = self.data.squeeze(0).chunk(2, dim=-1)

		plt.plot(pos, label='pos')
		plt.plot(vel, label='vel')
		plt.legend()
		plt.grid()
		plt.show()


class MLMD_VariableTrajectorySegment_DataSet(MD_DataSet):

	def __init__(self, config, segments):

		MD_DataSet.__init__(self, config)

		assert type(segments)==list
		for segment in segments:
			assert type(segment)==torch.Tensor
			assert segment.dim()==2

		self.segments = copy.deepcopy(segments)

	def load_and_process_data(self):

		pass

	def setup(self, stage: Optional[str] = None):

		self.load_and_process_data()

		'''
		val_split: point in timeseries from where we consider it the validation trajectory
		'''

		if 0<self.config.val_split:
			if len(self.segments)==1: # mainly for the initial trajectory
				assert self.segments[0].shape[0] > self.config.output_length_train+self.config.output_length_val
				T = self.segments[0].shape[0]
				train_segment, val_segment = self.segments[0].split([int(self.config.val_split*T), T - int(self.config.val_split * T)])
				assert train_segment.shape[0] >= self.config.output_length_train, f"{train_segment.shape[0]} !>= {self.config.output_length_train}"
				assert val_segment.shape[0] >= self.config.output_length_val, f"{val_segment.shape[0]} !>= {self.config.output_length_val}"
				train_segments, val_segments = [train_segment], [val_segment]
			elif len(self.segments)>=2:
				train_data_length = torch.cat(self.segments, dim=0).shape[0]
				train_length, val_length = int(self.config.val_split * train_data_length), int((1-self.config.val_split) * train_data_length)

				train_segments, val_segments = [self.segments[0]], [self.segments[-1]]
				for i, segment in enumerate(reversed(self.segments[1:-1])): # go reverse over list [ seq2, seq3, seq4, ...seqN-1 ]
					if torch.cat(val_segments, dim=0).shape[0]<val_length:
						 val_segments += [segment]
					else:
						train_segments += [segment]
		elif self.config.val_split==0:

			warnings.warn(f"Val Split = 0: Training and validation data set are the same")
			warnings.warn(f"Overwriting self.output_lengt_train")

			train_segments = copy.deepcopy(self.segments)
			val_segments = copy.deepcopy(self.segments)
			self.config.output_length_train = min(min([segment.shape[0] for segment in self.segments]), int(2*np.pi/self.config.dt))
			self.config.output_length_train -= 2 if 'bi' in self.config.model else 1
			self.config.output_length_val = self.config.output_length_train

		for train_segment in train_segments: assert train_segment.shape[0]>=self.config.output_length_train
		for val_segment in val_segments: assert val_segment.shape[0]>=self.config.output_length_val

		'''
		data.shape = [ts, timesteps, features]
		'''
		self.data_mean = torch.cat(train_segments, dim=0).mean(dim=[0])
		self.data_std = torch.cat(train_segments, dim=0).std(dim=[0])

		for train_segment in train_segments: 	train_segment.sub_(self.data_mean).div_(self.data_std)
		for val_segment in val_segments:        val_segment.sub_(self.data_mean).div_(self.data_std)

		self.train_segments = train_segments
		self.val_segments = val_segments

		if 'bi' in self.config.model:
			self.train_dataset = BiDirectional_VariableTimeSeries_DataSet(seqs=self.train_segments,
									input_length=self.config.input_length,
									output_length=self.config.output_length_train,
									traj_repetition=self.config.train_traj_repetition,
									)
			self.val_dataset = BiDirectional_VariableTimeSeries_DataSet(seqs=self.val_segments,
								      input_length=self.config.input_length,
								      output_length=self.config.output_length_val,
								      traj_repetition=self.config.train_traj_repetition,
								      )

		else:  # the unidirectional case
			self.train_dataset = VariableTimeSeries_DataSet(seqs=self.train_segments,
								     input_length=self.config.input_length,
								     output_length=self.config.output_length_train,
								     traj_repetition=self.config.train_traj_repetition,
								     )
			self.val_dataset = VariableTimeSeries_DataSet(seqs=self.val_segments,
								     input_length=self.config.input_length,
								     output_length=self.config.output_length_val,
								     traj_repetition=self.config.train_traj_repetition,
								     )

		print(f"Train DataSet: {self.train_dataset} SimMD Data")
		print(f"Val DataSet: {self.val_dataset} SimMD Data")

	def train_dataloader(self, *args, **kwargs) -> DataLoader:
		weights = [data_.shape[0] for data_ in self.train_segments]
		num_samples = len(self.train_dataset)
		weighted_sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples)
		dataloader = DataLoader(self.train_dataset, batch_size=self.config.batch_size, sampler=weighted_sampler, num_workers=self.config.num_workers)

		return dataloader

	def val_dataloader(self, *args, **kwargs) -> DataLoader:
		weights = [data_.shape[0] for data_ in self.val_segments]
		num_samples = len(self.val_dataset)
		weighted_sampler = WeightedRandomSampler(weights=weights, num_samples=num_samples)
		dataloader = DataLoader(self.val_dataset, batch_size=self.config.batch_size, sampler=weighted_sampler, num_workers=self.config.num_workers)

		return dataloader

	def plot_traj(self):

		pos, vel = self.data.squeeze(0).chunk(2, dim=-1)

		plt.plot(pos, label='pos')
		plt.plot(vel, label='vel')
		plt.legend()
		plt.grid()
		plt.show()


class QuantumMachine_DFT(MD_DataSet):
	'''
	Benzene: 49862, 72
	'''
	def __init__(self, config):

		MD_DataSet.__init__(self, config)

	def prepare_data(self, *args, **kwargs):

		url = "http://www.quantum-machine.org/gdml/data/npz/" + self.config.dataset

		if not os.path.exists(self.data_path):
			print(f'Downloading {self.data_str} from quantum-machine.org/gdml/data/npz')
			urllib.request.urlretrieve(url, self.data_path)

	def load_and_process_data(self):

		# path_npz = "../data/" + self.data_str
		path_npz = Path(__file__).absolute().parent.parent / ('data/'+self.config.dataset)
		data = np.load(path_npz)

		try:	data = data['R']
		except:	print("Error preparing data")

		data = torch.from_numpy(data).float()
		self.raw_data = data

		pos = data[:-1]
		vel = (data[1:] - data[:-1])

		pos = pos.flatten(-2, -1).unsqueeze(0)
		vel = vel.flatten(-2, -1).unsqueeze(0)

		self.data = torch.cat([pos, vel], dim=-1)

		self.dims = (self.data.shape[-1],)
		self.dim = 1

		self.in_features = self.data.shape[-1]
		self.out_features = self.data.shape[-1]

	def save_as_npz(self, pred, true, conditions, name, path):
		path_npz = Path(__file__).absolute().parent.parent / ('data/' + self.config.dataset)
		ref_data_dict = np.load(path_npz, allow_pickle=True)

		num_atoms = pred.shape[-1]//(2*3) # Features =[ pos*3, vel*3]
		timesteps = pred.shape[0]

		pred = self.unnormalize(pred)
		true = self.unnormalize(true)

		pred_pos, pred_vel = pred.chunk(chunks=2, dim=-1)
		true_pos, true_vel = true.chunk(chunks=2, dim=-1)

		pred_pos = pred_pos.reshape(1, timesteps, num_atoms, 3)
		true_pos = true_pos.reshape(1, timesteps, num_atoms, 3)

		npz_dict = dict(ref_data_dict)

		npz_dict.update({'R': pred_pos.float().numpy()})
		np.savez(f"{path}/Pred_{name}", **npz_dict)

		npz_dict.update({'R': true_pos.float().numpy()})
		np.savez(f"{path}/True_{name}", **npz_dict)


class Keto_DFT(MD_DataSet):

	def __init__(self, config):
		MD_DataSet.__init__(self, config)

	def prepare_data(self, *args, **kwargs):

		pass

	def load_and_process_data(self):
		if self.data_str == 'keto_100K_0.2fs.npz':

			self.pos_path = 'data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.100K.CLMD.1B_DT-0.2FS_01.POSITION.0.npz'
			self.vel_path = 'data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.100K.CLMD.1B_DT-0.2FS_01.VELOCITIES.0.npz'

		elif self.data_str == 'keto_300K_0.2fs.npz':

			self.pos_path = 'data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.300K.CLMD.1B_DT-0.2FS_01.POSITION.0.npz'
			self.vel_path = 'data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.300K.CLMD.1B_DT-0.2FS_01.VELOCITIES.0.npz'

		elif self.data_str == 'keto_300K_1.0fs.npz':

			self.pos_path = 'data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.300K.CLMD.1B_DT-1.0FS_01.POSITION.0.npz'
			self.vel_path = 'data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.300K.CLMD.1B_DT-1.0FS_01.VELOCITIES.0.npz'

		elif self.data_str == 'keto_500K_0.2fs.npz':

			self.pos_path = 'data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.500K.CLMD.1B_DT-0.2FS_01.POSITION.0.npz'
			self.vel_path = 'data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.500K.CLMD.1B_DT-0.2FS_01.VELOCITIES.0.npz'

		self.pos_path = Path(__file__).absolute().parent.parent / self.pos_path
		self.vel_path = Path(__file__).absolute().parent.parent / self.vel_path

		pos = np.load(self.pos_path)['R']
		vel = np.load(self.vel_path)['V']

		pos = torch.from_numpy(pos).float()
		vel = torch.from_numpy(vel).float()

		pos = pos.flatten(-2, -1)
		vel = vel.flatten(-2, -1)

		self.data = torch.cat([pos, vel], dim=-1)
		self.dims = (self.data.shape[-1],)
		self.in_features = self.data.shape[-1]
		self.out_features = self.data.shape[-1]

	def save_prediction(self, pred_data, true_data=None):

		assert pred_data.dim()==2
		assert pred_data.shape[-1]==self.data.shape[-1]

		npz_ = np.load(self.pos_path, allow_pickle=True)
		num_atoms = npz_['typ'].size # 1D array of letters denoting the atoms
		if true_data is not None:
			assert true_data.dim() == 2
			assert true_data.shape[-1] == self.data.shape[-1]
			true_data = true_data * self.data_std + self.data_mean
			timesteps, _ = true_data.shape
			pos, vel = true_data.chunk(chunks=2, dim=-1)
			pos = pos.reshape(1, timesteps, num_atoms, 3)
			assert F.mse_loss(pos, torch.from_numpy(npz_['R'][:,:timesteps]),reduction='sum')<=1e-3, f"{F.mse_loss(pos, torch.from_numpy(npz_['R'][:, :timesteps]))=}"
			npz_dict = dict(npz_)
			npz_dict.update({'R': pos.float().numpy()})
			np.savez(self.pos_path[:-4] + '.VAL' + '.npz', **npz_dict)

		data = pred_data * self.data_std + self.data_mean
		pos, vel = data.chunk(chunks=2, dim=-1)
		timesteps, features = data.shape
		pos = pos.reshape(1, timesteps, num_atoms, 3)

		npz_dict = dict(npz_)
		npz_dict.update({'R': pos.float().numpy()})
		np.savez(self.pos_path[:-4]+'.VALPRED'+'.npz', **npz_dict)

	def save_as_npz(self, pred, y, conditions, name, path):
		path_npz = Path(__file__).absolute().parent.parent / ('data/' + self.config.dataset)
		ref_data_dict = np.load(self.pos_path , allow_pickle=True)

		num_atoms = pred.shape[-1] // (2 * 3)  # Features =[ pos*3, vel*3]
		timesteps = pred.shape[0]

		pred = self.unnormalize(pred)
		true = self.unnormalize(y)

		pred_pos, pred_vel = pred.chunk(chunks=2, dim=-1)
		true_pos, true_vel = true.chunk(chunks=2, dim=-1)

		pred_pos = pred_pos.reshape(1, timesteps, num_atoms, 3)
		true_pos = true_pos.reshape(1, timesteps, num_atoms, 3)

		npz_dict = dict(ref_data_dict)

		npz_dict.update({'R': pred_pos.float().numpy()})
		np.savez(f"{path}/Pred_{name}", **npz_dict)

		npz_dict.update({'R': true_pos.float().numpy()})
		np.savez(f"{path}/True_{name}", **npz_dict)


class PMatrix_DataSet(MD_DataSet):
	'''
	P Matrix: We exploit the symmetry by only working on the upper triangular elements
	'''

	dims = (13,13)
	in_features = int(dims[0]*(dims[0]+1)/2)
	out_features = int(dims[0]*(dims[0]+1)/2)
	dim = 2

	def __init__(self, config):

		self.config = config
		self.data_path = traverse_path(__file__, -2)+'/data/H2O/PMatrix/P_matrix.dat'
		self.data_str = config.dataset_nicestr

	def load_and_process_data(self):

		print(f"{self.data_path=}")
		self.raw_data = torch.from_numpy(np.loadtxt(self.data_path)).double()
		assert self.raw_data.dim()==2
		T = self.raw_data.shape[0]
		square_dim = int(self.raw_data.shape[1] ** (0.5))
		data = self.raw_data.reshape(-1, square_dim, square_dim)
		triu_indices = torch.triu_indices(square_dim, square_dim)
		data = data[:, triu_indices[0], triu_indices[1]]
		assert data.dim()==2 and data.shape[1]==square_dim*(square_dim+1)/2==self.in_features==self.out_features
		self.data = data.unsqueeze(0).float()
		assert self.data.dim()==3

	def setup(self, stage: Optional[str] = None):

		self.load_and_process_data()

		self.data_mean = self.data.mean(dim=[0, 1])
		self.data_std = self.data.std(dim=[0, 1])

		self.data_norm = (self.data - self.data_mean) / (self.data_std + 1e-8)

		train_data_size = int(self.data.shape[1] * self.config.val_split * self.config.pct_dataset)
		val_data_size = int(self.data.shape[1] * (1 - self.config.val_split))

		'''
		val_split: point in timeseries from where we consider it the validation trajectory
		'''
		val_split = int(self.data_norm.shape[1] * self.config.val_split)
		data_train = self.data_norm[:, :int(val_split * self.config.pct_dataset)]  # future reduce val_split by training percentage
		data_val = self.data_norm[:, int(val_split * self.config.pct_dataset):int(val_split * self.config.pct_dataset + val_data_size)]

		self.y_mu = data_train.data.mean(dim=[0, 1])
		self.y_std = data_train.data.std(dim=[0, 1])

		self.dy = (data_train.data[:, 2:, :] - data_train.data[:, :-2, :]) / 2
		self.dy_mu = self.dy.mean(dim=[0, 1]).unsqueeze(0)  # shape = [bs=1, f]
		self.dy_std = self.dy.std(dim=[0, 1]).unsqueeze(0)  # shape = [bs=1, f]

		if 'bi' in self.config.model:
			self.data_train = BiDirectional_TimeSeries_DataSet(data=data_train,
									   input_length=self.config.input_length,
									   output_length=self.config.output_length_train,
									   output_length_sampling=self.config.output_length_sampling,
									   traj_repetition=self.config.train_traj_repetition,
									   sample_axis='timesteps')
			self.data_val = BiDirectional_TimeSeries_DataSet(data=data_val,
									 input_length=self.config.input_length,
									 output_length=self.config.output_length_val,
									 output_length_sampling=self.config.output_length_sampling,
									 traj_repetition=1,
									 sample_axis='timesteps')

		else:  # the unidirectional case
			self.data_train = TimeSeries_DataSet(data=data_train,
							     input_length=self.config.input_length,
							     output_length=self.config.output_length_train,
							     output_length_sampling=self.config.output_length_sampling,
							     traj_repetition=self.config.train_traj_repetition,
							     sample_axis='timesteps')
			self.data_val = TimeSeries_DataSet(data=data_val,
							   input_length=self.config.input_length,
							   output_length=self.config.output_length_val,
							   output_length_sampling=self.config.output_length_sampling,
							   traj_repetition=1,
							   sample_axis='timesteps')

	@typechecked
	def reconstruct_matrix(self, data: TensorType['BS', 'T', 'F']):
		pass

	def __repr__(self):
		return super().__repr__()+f' -> Triangular Elements of {torch.Size([*self.data.shape[:2], *self.dims])}'


class PSMatrix_DataSet(MD_DataSet):
	'''
	PS Matrix: Unfortunately not entirely symmetric
	'''

	dims = (13,13)
	in_features = math.prod(dims)
	out_features = math.prod(dims)
	dim = 2

	def __init__(self, config):

		self.config = config
		self.data_path = traverse_path(cwd, -1)+'/data/H2O/PMatrix/PS_matrix.dat'
		self.data_str = config.dataset_nicestr

	def CarParrinelloMD(self):

		'''
		coeff_m = (-1)^(m+1) * m * binom(2K, K-m) / binom(2K-2, K-1)
		L_K = sum_m^K (-1)^(m+1) * m * binom(2K, K-m) / binom(2K-2, K-1) P_{n-m} @ S_{n-m}
		P_n = L_K @ P_{n-1} @ L.T

		With K=4 and n=10
		C_n 	= { coeff(1)*P_9 @ S_9 + coeff(2)*P_8 @ S_8 + coeff(3)*P_7 @ S_7 + coeff(4)*P_6 @ S_6 } @ C_9
			= { coeff(4)*P_6 @ S_6 + coeff(3)*P_7 @ S_7 + coeff(2)*P_8 @ S_8 + coeff(1)*P_9 @ S_9 } @ C_9

		'''

		K = 10
		coeffs = torch.Tensor([38/11, -57/11,
				       684/143, -456/143, 228/143,
				       -171/286,
				       399/2431, -76/2431, 9/2431,
				       -1/4862])
		coeffs = coeffs[:K]
		'''
		coeffs are for m = [ 1, 2, 3, ... K ]
		we need to flip them in order to align them in the right order with P_{n-m} @ S_{n-m} 
		'''
		coeffs = coeffs.flip(dims=[0])

		P = torch.from_numpy(np.loadtxt(traverse_path(os.getcwd(), -1) + '/data/H2O/PMatrix/P_matrix.dat')).double()
		S = torch.from_numpy(np.loadtxt(traverse_path(os.getcwd(), -1) + '/data/H2O/PMatrix/S_matrix.dat')).double()
		assert P.shape==S.shape
		assert P.dim()==2

		square_dim = int(P.shape[1]**0.5)
		P = P.reshape(P.shape[0], square_dim, square_dim)
		S = S.reshape(S.shape[0], square_dim, square_dim)

		assert P.dim()==3
		assert (P - P.permute(0,2,1)).abs().sum()==0, f'{(P - P.permute(0, 2, 1)).abs().sum()=}'

		PS = torch.bmm(P,S)
		assert PS.shape==P.shape==S.shape

		'''
		[[ PS_0, PS_1, ... , PS_{K-1}], [ PS_1, PS_2, ... , PS_K], ... ]
		'''
		PS_unfolded = PS.unfold(dimension=0, size=K, step=1).permute(0,3,1,2) # [1001, 13, 13] -> [992, 10, 13, 13]
		# print(f"{P.shape=} -> {PS_unfolded.shape=}")
		assert torch.allclose(PS_unfolded[0],PS[:K])
		assert torch.allclose(PS_unfolded[1],PS[1:K+1])

		coeffs = coeffs.reshape(1,-1,1,1) # [m] -> [T=1, K, 1, 1] where the last two dims are for the matrix

		L = (coeffs * PS_unfolded).sum(dim=1)
		assert L.dim()==P.dim()==3
		'''
		With
		K = 4 and n = 10
		C_10 	= {coeff(1) * P_9 @ S_9 + coeff(2) * P_8 @ S_8 + coeff(3) * P_7 @ S_7 + coeff(4) * P_6 @ S_6} @ C_9
			= {coeff(4) * P_6 @ S_6 + coeff(3) * P_7 @ S_7 + coeff(2) * P_8 @ S_8 + coeff(1) * P_9 @ S_9} @ C_9
		First L incorporates elements up to K with the Kth entry being the last,
		Consequentially, P has to start at K
		'''
		T = 900
		P_batchpred = (L[:T]).bmm(P[K:T+K]).bmm(L[:T].permute(0,2,1))/4

		#################### Simple example
		P = torch.Tensor([[2,-2,-4],[-1,3,4],[1,-2,-3.1]]).unsqueeze(0)
		print(P.bmm(P))
		for _ in range(10):
			P = 3*torch.matrix_power(P,2) - 2*torch.matrix_power(P,3)
		print(f"{(P.bmm(P) - P).sum() =}")
		################### End Simple example

		print(f"{(P.bmm(S).bmm(P) - 2*P).sum() =}")
		print(f"{(P_batchpred.bmm(S[K:T+K]).bmm(P_batchpred) - 2*P_batchpred).sum() =}")
		print(f"{(P_batchpred.bmm(P_batchpred) - P_batchpred).sum() =}")

		# print(f"{(P_batchpred.bmm(S[K:T+K]).bmm(P_batchpred) - 2*P_batchpred).sum() =}") #dividing by four
		assert (S[0] - S[0].T).abs().sum()==0
		S_root = torch.linalg.cholesky(S[0])
		P_ = P[0]
		for _ in range(3):
			P_ = P_ @ S_root
			# P_ = 3*torch.matrix_power(P_, 2) - 2 * torch.matrix_power(P_,3)
			P_ = torch.matrix_power(P_, 2)@(3*torch.eye(P_.shape[0]) - 2 * P_ )
			print(f"{torch.trace(P_.mm(S[0]).mm(P_) - 2*P_) =}")

		exit()


		'''
		Autoregressive Implementation
		P_AR: P[0, 1, 2, ..., K-1]
		'''
		# P_AR = P[:K] # take first K matrices
		# coeffs.squeeze_(0) # shape = [10,1,1]
		# for n in range(K,P.shape[0]): # K=10/t=9 -> n=[ 10, 11, 12, 13, 14 ]
		#
		# 	'''
		# 	L_K = sum_{m=1}^K coeffs_m * P_{n-m} @ S_{n-m}
		# 	'''
		# 	assert coeffs.dim()==P.dim()==S.dim()==3
		# 	assert torch.allclose(P[n - K:n].bmm(S[n - K:n]), PS[n-K:n]), f"{n=}"
		# 	L_K = (coeffs * P[n-K:n].bmm(S[n-K:n])) # coeffs * P_{n-m} @ P_{n-m}
		# 	assert L_K.shape[0]==K
		# 	L_K = L_K.sum(dim=0)
		# 	assert torch.allclose(L_K, L[n-K]), f"step:{n}" # The precomputed L only starts at timestep 0
		# 	assert L_K.dim()==P[n].dim()==2
		#
		# 	P_n = L_K @ P[n] @ (L_K.T)
		# 	# P_n = L_K.mm(P_AR[-1].mm(L_K.T))
		# 	assert torch.allclose(L_K.mm(P_AR[n].mm(L_K.T)), L[n] @ P[n] @ (L[n].T)), f"{n=}"
		# 	assert P_n.dim()==2
		# 	assert torch.allclose(P_n/4, P_batchpred[n-K]), f"{n=}: {(P_n/4 - P_batchpred[n]).abs().sum()=}"
		# 	# for k in range(3): P_n = P_n.mm(P_n).mm(3-2*P_n)
		# 	P_AR = torch.cat([P_AR, P_n.unsqueeze(0)], dim=0)
		#
		# 	# P_ = torch.cat([P_, P__.unsqueeze(0)], dim=0)
		#
		# P = P[K:]
		# P_AR = P_AR[K:]
		# # P_batchpred
		# print(f"{(P_AR.bmm(S[K:]).bmm(P_AR) - 2*P_AR).mean() =}")
		#
		# assert P.dim()==P_AR.dim()==3, f'{P.dim()=} VS {P_AR.dim()=} VS 3'

		T_max = 50
		plt.plot(P.flatten(-2,-1)[K:T_max+K,0], label='True', color='b', alpha=0.5)
		plt.plot(P.flatten(-2,-1)[K:T_max+K,0:], color='b', alpha=0.5)
		plt.plot(P_batchpred.flatten(-2,-1)[:T_max,0], label='Pred AR', color='g', alpha=0.5)
		plt.plot(P_batchpred.flatten(-2,-1)[:T_max,0:], color='g', alpha=0.5)
		plt.legend()
		plt.ylim(-0.3,1)
		plt.show()

		exit('exited in MD Car Parinello MD')

	def load_and_process_data(self):

		'''
		Triangularize matrix
		@return:
		'''

		self.raw_data = torch.from_numpy(np.loadtxt(self.data_path)).float()
		assert self.raw_data.dim()==2
		T = self.raw_data.shape[0]
		self.data = copy.deepcopy(self.raw_data).unsqueeze(0)
		assert self.data.dim()==3, f'{self.data.shape=}'

	def setup(self, stage: Optional[str] = None):

		self.load_and_process_data()

		assert self.data.dim()==3

		self.data_mean = self.data.mean(dim=[0, 1])
		self.data_std = self.data.std(dim=[0, 1])

		self.data_norm = (self.data - self.data_mean) / (self.data_std + 1e-8)

		Seqs, T, F = self.data.shape

		train_data_size = int(T * self.config.val_split * self.config.pct_dataset)
		val_data_size = int(T * (1 - self.config.val_split))

		'''
		val_split: point in timeseries from where we consider it the validation trajectory
		'''
		val_split = int(T * self.config.val_split)
		data_train = self.data_norm[:, :int(val_split * self.config.pct_dataset)]  # future reduce val_split by training percentage
		data_val = self.data_norm[:, int(val_split * self.config.pct_dataset):int(val_split * self.config.pct_dataset + val_data_size)]

		self.y_mu = data_train.data.mean(dim=[0, 1])
		self.y_std = data_train.data.std(dim=[0, 1])

		self.dy = (data_train.data[:, 2:, :] - data_train.data[:, :-2, :]) / 2
		self.dy_mu = self.dy.mean(dim=[0, 1]).unsqueeze(0)  # shape = [bs, f]
		self.dy_std = self.dy.std(dim=[0, 1]).unsqueeze(0)  # shape = [bs, f]

		if 'bi' in self.config.model:
			self.data_train = BiDirectional_TimeSeries_DataSet(data=data_train,
									   input_length=self.config.input_length,
									   output_length=self.config.output_length_train,
									   output_length_sampling=self.config.output_length_sampling,
									   traj_repetition=self.config.train_traj_repetition,
									   sample_axis='timesteps')
			self.data_val = BiDirectional_TimeSeries_DataSet(data=data_val,
									 input_length=self.config.input_length,
									 output_length=self.config.output_length_val,
									 output_length_sampling=self.config.output_length_sampling,
									 traj_repetition=1,
									 sample_axis='timesteps')

		else:  # the unidirectional case
			self.data_train = TimeSeries_DataSet(data=data_train,
							     input_length=self.config.input_length,
							     output_length=self.config.output_length_train,
							     output_length_sampling=self.config.output_length_sampling,
							     traj_repetition=self.config.train_traj_repetition,
							     sample_axis='timesteps')
			self.data_val = TimeSeries_DataSet(data=data_val,
							   input_length=self.config.input_length,
							   output_length=self.config.output_length_val,
							   output_length_sampling=self.config.output_length_sampling,
							   traj_repetition=1,
							   sample_axis='timesteps')

	def plot_sequential_prediction(self, y: list, pred: list, y0=None, t0=None, num_sequential_samples=None, extra_title='', extra_text=''):
		'''
		pred: the prediction by the neural network
		'''

		assert isinstance(y, list)
		# assert isinstance(y0, torch.Tensor)
		assert isinstance(pred, list)
		if y0 is not None: assert isinstance(y0, torch.Tensor), f"{type(y0)=}"

		assert y[0].dim() == pred[0].dim() == 2

		colors = ['r', 'g', 'b']

		fig, axs = plt.subplots(1, 1, sharex=True, figsize=(30, 10))
		# axs = axs.flatten()

		num_plots = y[0].shape[-1] // 2  # determine how many pos and vel plots there are
		if y0 is not None and t0 is not None: t_y0 = t0[:y0.shape[0]]

		for plot_i in [0, 1, 2][:num_plots]:  # plot the first three at most which are the first three position timeseries
			t0_ = 0
			for y_i, pred_i in zip(y, pred):
				t_i = t0_ + torch.arange(y_i.shape[0])
				axs.plot(t_i, pred_i[:, plot_i], ls='--', color=colors[plot_i], label='Data' if plot_i == -1 else None)
				t0_ += y_i.shape[0]
				print(f"{t0_=}")
				if t0_ > t_y0.max(): break
				if num_sequential_samples is not None:
					if t0_ > num_sequential_samples: break

			if y0 is not None and t0 is not None: axs.scatter(t_y0, y0[:t_y0.shape[0], plot_i], color=colors[plot_i], s=150, marker='|', label='Initial and Final Conditions' if plot_i == -1 else None)

			axs.plot(torch.cat(y)[:t_y0.max(), plot_i], ls='-', color=colors[plot_i], label='Prediction' if plot_i == -1 else None)

		axs.set_xlabel('Steps')
		axs.set_ylabel('$q(t)$')
		axs.grid()
		axs.set_ylim(-3 if min([pred_.min() for pred_ in pred]) < -10 else None, 10 if min([pred_.max() for pred_ in pred]) > 10 else None)

		fig.suptitle(self.config.dataset_nicestr + extra_title)
		plt.legend()

		if extra_text is not None:
			bbox = dict(facecolor='white')
			axs.text(1.01, 0.98, extra_text, transform=axs.transAxes, va='top', bbox=bbox)
		plt.tight_layout()

		plt.show()

		return fig

	@typechecked
	def reconstruct_matrix(self, data: TensorType['BS', 'T', 'F']):
		assert data.shape[-1]==math.prod(self.dims)
		return data.reshape(*data.shape[:2], *self.dims)


class HMC_DM(MD_DataSet):

	def __init__(self, config):

		self.config = config
		self.batch_size = self.config.batch_size
		self.data_str = 'HMC'

		super().__init__(config)

	def setup(self, *args, **kwargs):

		# from DiffEqNets.MolecularDynamics.data.HMC_Data_Generation import HMCData
		from MLMD.data.HMC_Data_Generation import HMCData

		num_datasets = 1
		num_trajectories = 500
		num_means = 5
		trajectory_stepsize = 0.2
		dist_mean_min_max = 4
		num_steps = 800

		means, covars, surfaces, trajectories = [], [], [], []
		for i_dataset in range(num_datasets):
			data_gen = HMCData(num_trajs=num_trajectories, min_max=dist_mean_min_max, num_steps=num_steps,
					   num_means=num_means, step_size=trajectory_stepsize)
			surface, X_grid, Y_grid = data_gen.generate_plottingsurface()
			surface = (surface - surface.min()) / (surface.max() - surface.min())
			trajs = data_gen.generate_trajectory()
			if False:
				data_gen.plot_surface_and_trajectories()
			# data_gen.plot
			surface = torch.from_numpy(surface).unsqueeze(0).repeat(num_trajectories, 1, 1).unsqueeze(0)

			mean = data_gen.potential.means.unsqueeze(0).repeat(num_trajectories, 1, 1).unsqueeze(0)
			covar = data_gen.potential.covars.unsqueeze(0).repeat(num_trajectories, 1, 1, 1).unsqueeze(0)


			# exit()
			means.append(mean)
			covars.append(covar)

			surfaces.append(surface)
			trajectories.append(torch.from_numpy(trajs).unsqueeze(0))

			X_grid = data_gen.X_grid
			Y_grid = data_gen.Y_grid

		self.data = torch.cat(trajectories, dim=0).squeeze(0)
		self.surfaces = torch.cat(surfaces, dim=0)
		self.means = torch.cat(means, dim=0)
		self.covars = torch.cat(covars, dim=0)

		assert self.data.dim() == 3
		assert self.data.shape==(num_trajectories, num_steps+1, 4), f"{self.data.shape=}"

		self.data_norm = (self.data - self.data.mean(dim=[0, 1])) / (self.data.std(dim=[0, 1]) + 1e-3)

		val_split = int(self.data.shape[1] * self.config.val_split)
		data_train, data_val = self.data_norm[:, :val_split], self.data_norm[:, val_split:]

		self.y_mu = data_train.mean(dim=[0, 1])
		self.y_std = data_train.std(dim=[0, 1])

		self.dy = (data_train[:, 2:, :] - data_train[:, :-2, :]) / 2
		self.dy_mu = self.dy.mean(dim=[0, 1]).unsqueeze(0)  # shape = [bs, f]
		self.dy_std = self.dy.std(dim=[0, 1]).unsqueeze(0)  # shape = [bs, f]


		if 'bi' in self.config.model:
			self.data_train = BiDirectional_TimeSeries_DataSet(data=data_train,
									   input_length=self.config.input_length,
									   output_length=self.config.output_length_train,
									   traj_repetition=self.config.train_traj_repetition,
									   sample_axis='trajs')
			self.data_val = BiDirectional_TimeSeries_DataSet(data=data_val,
									 input_length=self.config.input_length,
									 output_length=self.config.output_length_val,
									 traj_repetition=self.config.train_traj_repetition,
									 sample_axis='trajs')

		else:  # the unidirectional case
			self.data_train = TimeSeries_DataSet(data=data_train,
							     input_length=self.config.input_length,
							     output_length=self.config.output_length_train,
							     traj_repetition=self.config.train_traj_repetition)
			self.data_val = TimeSeries_DataSet(data=data_val,
							   input_length=self.config.input_length,
							   output_length=self.config.output_length_val,
							   traj_repetition=self.config.train_traj_repetition)

	def train_dataloader(self, *args, **kwargs) -> DataLoader:
		# print(f" @train_dataloader(): {self.data_train.data.shape=}")
		return DataLoader(self.data_train, batch_size=self.batch_size, shuffle=True)

	def val_dataloader(self, *args, **kwargs) -> DataLoader:
		return DataLoader(self.data_val, batch_size=self.batch_size, shuffle=True)

	def plot_sequential_prediction(self, y, y0, t0, pred):
		'''
		pred: the prediction by the neural network
		'''

		colors = ['r', 'g', 'b']

		fig = plt.figure(figsize=(30, 10))
		num_sequential_samples = 500

		plt.plot(y[:num_sequential_samples, -1], ls='-', color=colors[0], label='Data')
		plt.plot(y[:num_sequential_samples, -2], ls='-', color=colors[1])
		plt.plot(pred[:num_sequential_samples, -1], ls='--', color=colors[0], label='Prediction')
		plt.plot(pred[:num_sequential_samples, -2], ls='--', color=colors[1])

		t_y0 = t0[:y0.shape[0]]

		plt.scatter(t_y0, y0[:t_y0.shape[0], -1], color=colors[0], label='Initial and Final Conditions')
		plt.scatter(t_y0, y0[:t_y0.shape[0], -2], color=colors[1])

		plt.xlabel('t')
		plt.ylabel('$q(t)$')
		plt.title(self.config.dataset)
		plt.grid()
		# plt.xticks(np.arange(0,t_y0.max()))
		plt.xlim(0, t_y0.max())
		plt.legend()

		return fig

	def __repr__(self):
		return f'{self.data_str}: {self.data.shape} features'


class MLMD_Trajectory(object):

	def __init__(self, config: Dict):

		self.config = config
		self.traj = None 			# Tensor: [1, T, F]
		self.traj_type = None			# Tensor: [T] ∈ {0,1}
		self.traj_type_dict = {'mlmd':1, 'simmd':0}
		self.traj_segment_lengths = None	# Tensor: used to find indices to split the trajectory

	def __len__(self):
		return self.traj.shape[0]

	def __iadd__(self, other: Tuple) -> None:
		'''

		@param other: Tuple or list with (traj_segment: torch.Tensor, model_id: Union('simmd', 'mlmd'))
		@return:
		'''

		assert type(other) in [tuple, list]
		assert len(other)==2
		traj_segment, traj_type = other

		if traj_segment.dim()==3:
			assert traj_segment.shape[0]==1
			traj_segment.squeeze_(0)
		assert traj_segment.dim()==2
		assert traj_type in ['simmd', 'mlmd']

		if self.traj is None and self.traj_type is None and self.traj_segment_lengths is None:
			'''
			Init	self.traj as 2d tensor with trajectory data
				self.traj_type as 1d tensor with [0,1]'s where 1==mlmd and 0==simmd
				self.trajJ_segment_lengths as 1d int tensor with lengths of individual segments, to be used with torch.split 
			'''
			assert traj_segment.dim()==2
			self.traj = traj_segment
			self.traj_type = torch.ones(traj_segment.shape[0]) if traj_type=='mlmd' else torch.zeros(traj_segment.shape[0])
			self.traj_segment_lengths = Tensor([traj_segment.shape[0]]).int()
		else:
			'''
			1) Concat traj_segment to traj: traj = [traj, traj_segment]
			2) Create type indicator ∈ [0,1] for each timestep whether its mlmd<=>1 or simmd<=>0
			3) If type is same as last type, just increment traj_segment_lengths, else concat traj_segment_length
			'''
			self.traj = torch.cat([self.traj, traj_segment], dim=0)
			traj_type = torch.ones(traj_segment.shape[0]) if traj_type == 'mlmd' else torch.zeros(traj_segment.shape[0])

			assert self.traj_type.dim()==1 and traj_type.dim()==1, f'{self.traj_type.dim()=} or {self.traj_type.dim()=} are not of dim=1'
			if traj_type[0] != self.traj_type[-1]:
				self.traj_segment_lengths = torch.cat([self.traj_segment_lengths, Tensor([traj_segment.shape[0]]).int()])
			elif traj_type[0] == self.traj_type[-1]:
				self.traj_segment_lengths[-1].add_(traj_segment.shape[0])

			self.traj_type = torch.cat([self.traj_type, traj_type]) # after checking whether traj_segment is equal to last traj_segment_type, then add self.traj_type

		assert self.traj.dim()==2
		assert self.traj_type.dim()==1
		assert self.traj_segment_lengths.dim()==1
		assert self.traj.shape[0]==self.traj_type.shape[0]==self.traj_segment_lengths.sum()

		return self

	def get_simmd_traj_segments(self) -> List:
		'''
		Filter list of traj_segments for SimMD data
		Split each traj_segment into either train or validation length depending on which is larger
		'''
		traj = self.traj.split(self.traj_segment_lengths.tolist())
		traj_type = self.traj_type.split(self.traj_segment_lengths.tolist())

		assert len(traj)==len(traj_type)

		traj = list(traj)
		filtered_traj = []
		for i, segment_type in enumerate(traj_type):
			assert type(segment_type)==torch.Tensor
			assert torch.sum(segment_type - segment_type[0])==0. # Check that all entries in segment_type are the same number
			'''1: mlmd , 0: simmd -> if segment_type is mlmd, delete from trajs'''
			if segment_type[0]==self.traj_type_dict['simmd']: filtered_traj += [traj[i]]

		assert len(filtered_traj)>=1

		return filtered_traj

	def get_mlmd_traj_segments(self) -> List:
		'''
		Filter list of traj_segments for SimMD data
		Split each traj_segment into either train or validation length depending on which is larger
		'''

		traj = self.traj.split(self.traj_segment_lengths.tolist())
		traj_type = self.traj_type.split(self.traj_segment_lengths.tolist())

		assert len(traj) == len(traj_type)

		traj = list(traj)
		filtered_traj = []
		for i, segment_type in enumerate(traj_type):
			assert type(segment_type) == torch.Tensor
			assert torch.sum(segment_type - segment_type[0]) == 0.  # Check that all entries in segment_type are the same number
			'''1: mlmd , 0: simmd -> if segment_type is simmd, delete from trajs'''
			if segment_type[0] == self.traj_type_dict['simmd']: filtered_traj += [traj[i]]

		return filtered_traj

	def plot_simmd_traj_segments(self):
		plt.title('SimMD Data')

		simmd_data = torch.cat(self.get_simmd_traj_segments(), dim=0)
		plt.plot(simmd_data[:, 0], label='q(t)')
		plt.plot(simmd_data[:, 1], label='p(t)')
		plt.legend()
		plt.grid()
		plt.show()

	def remove_last_mlmd_forecast(self):

		assert self.traj_type[-1]==self.traj_type_dict['mlmd'], f"Trying to remove most recent MLMD forecast, but last traj_segment_id is not mlmd but {self.traj_segments_id[-1]}"

		self.traj 			= self.traj[:-self.traj_segment_lengths[-1]]
		self.traj_type 			= self.traj_type[:-self.traj_segment_lengths[-1]]
		self.traj_segment_lengths 	= self.traj_segment_lengths[:-1]

		assert self.traj.shape[0]==self.traj_type.shape[0]==self.traj_segment_lengths.sum()

	def __repr__(self):
		return f"MLMD Trajectory:: T:{len(self)}, F:{self.traj.shape[-1]} "


def load_dm_data(config):
	'''
	Creates the PyTorch Lightning DataModule and updates the hyperparameters
	'''
	data_str = config.dataset
	if data_str in ['benzene_dft.npz',
			'ethanol_dft.npz',
			'malonaldehyde_dft.npz',
			'toluene_dft.npz',
			'salicylic_dft.npz',
			'naphthalene_dft.npz',
			'paracetamol_dft.npz',
			'aspirin_dft.npz',
			'uracil_dft.npz']:

		dm = QuantumMachine_DFT(config)

	elif data_str in ['keto_100K_0.2fs.npz', 'keto_300K_0.2fs.npz', 'keto_300K_1.0fs.npz', 'keto_500K_0.2fs.npz']:

		dm = Keto_DFT(config)

	elif data_str in ['hmc']:
		dm = HMC_DM(config)
	elif data_str in ['p_matrix']:
		dm = PMatrix_DataSet(config)
	elif data_str in ['ps_matrix']:
		dm = PSMatrix_DataSet(config)
	else:
		exit(f"No valid dataset provided ...")

	dm.prepare_data()
	dm.setup()

	print(dm)

	assert hasattr(dm, 'dy_mu') and hasattr(dm, 'dy_std')
	config.__dict__.update({'in_dims': dm.dims})
	config.__dict__.update({'in_features': dm.in_features})
	config.__dict__.update({'out_features': dm.out_features})
	config.__dict__.update({'num_hidden': dm.in_features * config.num_hidden_multiplier})

	return dm


if __name__=='__main__':
	from MLMD.src.MD_HyperparameterParser import Interpolation_HParamParser

	config = Interpolation_HParamParser(dataset='ps_matrix', integration_mode='diffeq')
	dm = load_dm_data(config)

	dm.CarParrinelloMD()


