import os, shutil, inspect
from typing import List, Optional
from os import listdir
from os.path import isfile, join, isdir
import math, numpy as  np

import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt

import urllib.request

from ase import Atoms, Atom

from pytorch_lightning import LightningDataModule
from torch.utils.data import DataLoader, random_split
from torch.utils.data import Dataset, ConcatDataset
import torch.nn.functional as F

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def to_np(_tensor):
	if _tensor is not None:
		assert isinstance(_tensor, torch.Tensor)
		return _tensor.cpu().squeeze().detach().numpy()
	else:
		return None

class TimeSeries_DataSet(Dataset):

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
			idx = idx % self.data.shape[
				0]  # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
			traj = self.data[idx]  # selecting trajectory

			assert (traj.shape[0] - total_length) >= 0, f' trajectory length {traj.shape[0]} is larger than output {output_length}'
			t0 = np.random.choice(traj.shape[0] - total_length)  # selecting starting time in trajectory

		elif self.sample_axis == 'timesteps':
			'''
			Few short timeseries
			'''

			traj_index = np.random.choice(self.data.shape[0])  # Randomly select one of the few timeseries
			traj = self.data[traj_index]  # select the timeseries

			t0 = idx  # we're sampling from the timesteps

		y0 = traj[t0:(t0 + self.input_length)]  # selecting corresponding startin gpoint
		target = traj[t0:(t0 + total_length)]  # snippet of trajectory

		return y0, output_length, target

	def __len__(self):

		if self.sample_axis == 'trajs':
			return self.data.shape[0] * self.traj_repetition
		elif self.sample_axis == 'timesteps':
			return self.data.shape[1] - (self.input_length + self.output_length)
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

			t0 = idx  # we're sampling from the timesteps
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

		T = data.shape[0]
		last_part = T % (input_length + output_length)
		data = data[:(T - last_part)]
		assert data.dim() == 2, f'{data.shape=}'

		data_ = torch.stack(data.chunk(chunks=T // (input_length + output_length), dim=0)[:-1])  # dropping the last, possibly wrong length time series sample

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

	def __init__(self, hparams):

		self.data_path = '../data/'
		self.data_str = hparams.dataset
		self.hparams = hparams

		self.sequential_sampling=False

	def load_and_process_MD_data(self):

		raise NotImplementedError()

	def setup(self, stage: Optional[str] = None):

		self.load_and_process_MD_data()

		assert self.data.dim() == 3
		assert self.data.shape[0] == 1

		'''
		data.shape = [ts, timesteps, features]
		'''
		self.data_mean = self.data.mean(dim=[0,1])
		self.data_std = self.data.std(dim=[0,1])

		self.data_norm = (self.data - self.data_mean) / (self.data_std + 1e-8)

		train_data_size = int(self.data.shape[1] * self.hparams.val_split * self.hparams.pct_data_set)
		val_data_size = int(self.data.shape[1]*(1-self.hparams.val_split))

		'''
		val_split: point in timeseries from where we consider it the validation trajectory
		'''
		val_split = int(self.data_norm.shape[1] * self.hparams.val_split)
		data_train = self.data_norm[:, :int(val_split*self.hparams.pct_data_set)] # future reduce val_split by training percentage
		data_val = self.data_norm[:, int(val_split * self.hparams.pct_data_set):int(val_split * self.hparams.pct_data_set+val_data_size)]

		self.y_mu 	= data_train.data.mean(dim=[0, 1]).to(device)
		self.y_std 	= data_train.data.std(dim=[0, 1]).to(device)

		self.dy 	= (data_train.data[:, 2:, :] - data_train.data[:, :-2, :]) / 2
		self.dy_mu 	= self.dy.mean(dim=[0, 1]).unsqueeze(0).to(device)  # shape = [bs, f]
		self.dy_std 	= self.dy.std(dim=[0, 1]).unsqueeze(0).to(device)  # shape = [bs, f]

		if 'bi' in self.hparams.model:
			self.data_train = BiDirectional_TimeSeries_DataSet(data=data_train,
									   input_length=self.hparams.input_length,
									   output_length=self.hparams.output_length_train,
									   output_length_sampling=self.hparams.output_length_sampling,
									   traj_repetition=self.hparams.train_traj_repetition,
									   sample_axis='timesteps')
			self.data_val = BiDirectional_TimeSeries_DataSet(data=data_val,
									 input_length=self.hparams.input_length,
									 output_length=self.hparams.output_length_val,
									 output_length_sampling=self.hparams.output_length_sampling,
									 traj_repetition=self.hparams.train_traj_repetition,
									 sample_axis='timesteps')

		else:  # the unidirectional case
			self.data_train = TimeSeries_DataSet(data=data_train,
							     input_length=self.hparams.input_length,
							     output_length=self.hparams.output_length_train,
							     output_length_sampling=self.hparams.output_length_sampling,
							     traj_repetition=self.hparams.train_traj_repetition)
			self.data_val = TimeSeries_DataSet(data=data_val,
							   input_length=self.hparams.input_length,
							   output_length=self.hparams.output_length_val,
							   output_length_sampling=self.hparams.output_length_sampling,
							   traj_repetition=self.hparams.train_traj_repetition)

	def train_dataloader(self, *args, **kwargs) -> DataLoader:

		dataloader = DataLoader(self.data_train, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

		return dataloader

	def val_dataloader(self, *args, **kwargs) -> DataLoader:
		return DataLoader(self.data_val, batch_size=self.hparams.batch_size*2, num_workers=self.hparams.num_workers)

	def plot_sequential_prediction(self, y, y0, t0, pred):
		'''
		pred: the prediction by the neural network
		'''

		colors = ['r', 'g', 'b']

		fig = plt.figure(figsize=(30, 10))
		num_sequential_samples=500

		plt.plot(y[:num_sequential_samples, -3], ls='-', color=colors[0], label='Data')
		plt.plot(y[:num_sequential_samples, -2], ls='-', color=colors[1])
		plt.plot(y[:num_sequential_samples, -1], ls='-', color=colors[2])
		plt.plot(pred[:num_sequential_samples, -3], ls='--', color=colors[0], label='Prediction')
		plt.plot(pred[:num_sequential_samples, -2], ls='--', color=colors[1])
		plt.plot(pred[:num_sequential_samples, -1], ls='--', color=colors[2])

		t_y0 = t0[:y0.shape[0]]

		plt.scatter(t_y0, y0[:t_y0.shape[0], -3], color=colors[0], label='Initial and Final Conditions')
		plt.scatter(t_y0, y0[:t_y0.shape[0], -2], color=colors[1])
		plt.scatter(t_y0, y0[:t_y0.shape[0], -1], color=colors[2])
		plt.xlabel('t')
		plt.ylabel('$q(t)$')
		plt.title(self.hparams.dataset)
		plt.grid()
		# plt.xticks(np.arange(0,t_y0.max()))
		plt.xlim(0, t_y0.max())
		plt.legend()

		return fig

	def __repr__(self):

		return f'{self.data_str}: {self.data.shape} features'

class QuantumMachine_DFT(MD_DataSet):
	'''
	Benzene: 49862, 72
	'''

	def __init__(self, hparams):

		MD_DataSet.__init__(self, hparams)

	def prepare_data(self, *args, **kwargs):

		path_npz = "../data/" + self.data_str
		# path_pt = "../data/" + self.data_str.split(".")[0] + ".pt"

		url = "http://www.quantum-machine.org/gdml/data/npz/" + self.data_str

		if not os.path.exists('../data/' + self.data_str):
			print(f'Downloading {self.data_str} from quantum-machine.org/gdml/data/npz')
			urllib.request.urlretrieve(url, '../data/' + self.data_str)

	def load_and_process_MD_data(self):

		path_npz = "../data/" + self.data_str

		data = np.load(path_npz)
		try:
			data = data['R']

		except:
			print("Error preparing data")

		data = torch.from_numpy(data).float()

		pos = data[:-1]
		vel = (data[1:] - data[:-1])

		pos = pos.flatten(-2, -1).unsqueeze(0)
		vel = vel.flatten(-2, -1).unsqueeze(0)

		self.data = torch.cat([pos, vel], dim=-1)

class Keto_DFT(MD_DataSet):

	def __init__(self, hparams):

		self.data_path = '../data/'
		self.data_str = hparams.dataset
		self.hparams = hparams

	def prepare_data(self, *args, **kwargs):

		pass

	def load_and_process_MD_data(self):
		if self.data_str == 'keto_100K_0.2fs.npz':

			self.pos_path = '../data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.100K.CLMD.1B_DT-0.2FS_01.POSITION.0.npz'
			self.vel_path = '../data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.100K.CLMD.1B_DT-0.2FS_01.VELOCITIES.0.npz'

		elif self.data_str == 'keto_300K_0.2fs.npz':

			self.pos_path = '../data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.300K.CLMD.1B_DT-0.2FS_01.POSITION.0.npz'
			self.vel_path = '../data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.300K.CLMD.1B_DT-0.2FS_01.VELOCITIES.0.npz'

		elif self.data_str == 'keto_300K_1.0fs.npz':

			self.pos_path = '../data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.300K.CLMD.1B_DT-1.0FS_01.POSITION.0.npz'
			self.vel_path = '../data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.300K.CLMD.1B_DT-1.0FS_01.VELOCITIES.0.npz'

		elif self.data_str == 'keto_500K_0.2fs.npz':

			self.pos_path = '../data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.500K.CLMD.1B_DT-0.2FS_01.POSITION.0.npz'
			self.vel_path = '../data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.500K.CLMD.1B_DT-0.2FS_01.VELOCITIES.0.npz'

		pos = np.load(self.pos_path)['R']
		vel = np.load(self.vel_path)['V']

		pos = torch.from_numpy(pos).float()
		vel = torch.from_numpy(vel).float()

		pos = pos.flatten(-2, -1)
		vel = vel.flatten(-2, -1)

		self.data = torch.cat([pos, vel], dim=-1)

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

		# npz_['R'] = data * self.data_std + self.data_mean

class HMC_DM(MD_DataSet):

	def __init__(self, hparams):

		self.hparams = hparams
		self.batch_size = self.hparams.batch_size
		self.data_str = 'HMC'

		super().__init__(hparams)

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

			# print(mean.shape, covar.shape, surface.shape)
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

		val_split = int(self.data.shape[1] * self.hparams.val_split)
		data_train, data_val = self.data_norm[:, :val_split], self.data_norm[:, val_split:]

		self.y_mu = data_train.mean(dim=[0, 1]).to(device)
		self.y_std = data_train.std(dim=[0, 1]).to(device)

		self.dy = (data_train[:, 2:, :] - data_train[:, :-2, :]) / 2
		self.dy_mu = self.dy.mean(dim=[0, 1]).unsqueeze(0).to(device)  # shape = [bs, f]
		self.dy_std = self.dy.std(dim=[0, 1]).unsqueeze(0).to(device)  # shape = [bs, f]


		if 'bi' in self.hparams.model:
			self.data_train = BiDirectional_TimeSeries_DataSet(data=data_train,
									   input_length=self.hparams.input_length,
									   output_length=self.hparams.output_length_train,
									   traj_repetition=self.hparams.train_traj_repetition,
									   sample_axis='trajs')
			self.data_val = BiDirectional_TimeSeries_DataSet(data=data_val,
									 input_length=self.hparams.input_length,
									 output_length=self.hparams.output_length_val,
									 traj_repetition=self.hparams.train_traj_repetition,
									 sample_axis='trajs')

		else:  # the unidirectional case
			self.data_train = TimeSeries_DataSet(data=data_train,
							     input_length=self.hparams.input_length,
							     output_length=self.hparams.output_length_train,
							     traj_repetition=self.hparams.train_traj_repetition)
			self.data_val = TimeSeries_DataSet(data=data_val,
							   input_length=self.hparams.input_length,
							   output_length=self.hparams.output_length_val,
							   traj_repetition=self.hparams.train_traj_repetition)

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
		plt.title(self.hparams.dataset)
		plt.grid()
		# plt.xticks(np.arange(0,t_y0.max()))
		plt.xlim(0, t_y0.max())
		plt.legend()

		return fig

	def __repr__(self):
		return f'{self.data_str}: {self.data.shape} features'

def load_dm_data(hparams):
	data_str = hparams.dataset
	if data_str in ['benzene_dft.npz',
			'ethanol_dft.npz',
			'malonaldehyde_dft.npz',
			'toluene_dft.npz',
			'salicylic_dft.npz',
			'naphthalene_dft.npz',
			'paracetamol_dft.npz',
			'aspirin_dft.npz',
			'uracil_dft.npz']:

		dm = QuantumMachine_DFT(hparams)

	elif data_str in ['keto_100K_0.2fs.npz', 'keto_300K_0.2fs.npz', 'keto_300K_1.0fs.npz', 'keto_500K_0.2fs.npz']:

		dm = Keto_DFT(hparams)

	elif data_str in ['hmc']:
		dm = HMC_DM(hparams)
	else:
		exit(f"No valid dataset provided ...")

	dm.prepare_data()
	dm.setup()

	y_mu = dm.data_train.data.mean(dim=[0, 1])
	y_std = dm.data_train.data.std(dim=[0, 1])

	dy = (dm.data_train.data[:, 2:, :] - dm.data_train.data[:, :-2, :]) / 2

	dy_mu = dy.mean(dim=[0, 1]).unsqueeze(0)  # shape = [bs, f]
	dy_std = dy.std(dim=[0, 1]).unsqueeze(0)  # shape = [bs, f]

	print(dm)

	return dm


def plot_MD_data(hparams, batch_pred, batch_y, batch_val_pred, batch_val_y):
	val_plot_pred_pos = batch_val_pred[0, :, :6].cpu().detach().T
	val_plot_pred_vel = batch_val_pred[0, :, -6:].cpu().detach().T
	val_plot_pos = batch_val_y[0, :, :6].cpu().detach().T
	val_plot_vel = batch_val_y[0, :, -6:].cpu().detach().T

	train_plot_pred_pos = batch_pred[0, :, :6].cpu().detach().T
	train_plot_pred_vel = batch_pred[0, :, -6:].cpu().detach().T
	train_plot_pos = batch_y[0, :, :6].cpu().detach().T
	train_plot_vel = batch_y[0, :, -6:].cpu().detach().T

	cmap = plt.cm.get_cmap('viridis')
	colors = cmap(np.linspace(0, 1, val_plot_pos.shape[0]))

	fig, axs = plt.subplots(2, 2, sharex=True)
	plt.suptitle(hparams.logname + ' ' + hparams.model)
	axs = axs.flatten()
	axs[0].set_title('Val Pos')
	axs[1].set_title('Val Vel')
	axs[2].set_title('Train Pos')
	axs[3].set_title('Train Vel')
	for batch_i, color in enumerate(colors):
		# print(plot_pos.shape, plot_pred_pos.shape, batch_i)
		axs[0].plot(val_plot_pred_pos[batch_i], ls='--', color=color, alpha=0.5, label='pred')
		axs[0].plot(val_plot_pos[batch_i], color=color, alpha=0.5)
		axs[0].legend()
		# axs[0].set_ylim(val_plot_pred_pos[batch_i].min(), val_plot_pred_pos[batch_i].max())

		axs[1].plot(val_plot_pred_vel[batch_i], ls='--', color=color, alpha=0.5)
		axs[1].plot(val_plot_vel[batch_i], color=color, alpha=0.5)

		axs[2].plot(train_plot_pred_pos[batch_i], ls='--', color=color, alpha=0.5)
		axs[2].plot(train_plot_pos[batch_i], color=color, alpha=0.5)

		axs[3].plot(train_plot_pred_vel[batch_i], ls='--', color=color, alpha=0.5)
		axs[3].plot(train_plot_vel[batch_i], color=color, alpha=0.5)

		axs[2].set_ylim(train_plot_pos.min() * 3, train_plot_pos.max() * 3)
		axs[3].set_ylim(train_plot_vel.min() * 3, train_plot_vel.max() * 3)

	return fig


def plot_HMC_data(hparams, batch_pred, batch_y, batch_val_pred, batch_val_y):
	# print(batch_pred.shape)
	# exit()

	val_plot_pred_pos = batch_val_pred[:6, :, :2].cpu().detach()
	val_plot_pos = batch_val_y[:6, :, :2].cpu().detach()

	# print(val_plot_pred_pos.shape)
	# exit()

	train_plot_pred_pos = batch_pred[:6, :, :2].cpu().detach()
	train_plot_pos = batch_y[:6, :, :2].cpu().detach()

	fig, axs = plt.subplots(1, 2, sharex=True)
	plt.suptitle(hparams.logname + ' ' + hparams.model)
	axs = axs.flatten()

	cmap = plt.cm.get_cmap('viridis')
	colors = cmap(np.linspace(0, 1, val_plot_pos.shape[0]))

	for batch_i, color in enumerate(colors):
		# print(train_plot_pred_pos.shape, val_plot_pred_pos.shape, batch_i)
		# exit()
		axs[0].plot(val_plot_pred_pos[batch_i, :, 0], val_plot_pred_pos[batch_i, :, 1], ls='--', color=color, alpha=0.5, label='pred')
		axs[0].plot(val_plot_pos[batch_i, :, 0], val_plot_pos[batch_i, :, 1], color=color, alpha=0.5)
		axs[0].legend()
		# axs[0].set_ylim(val_plot_pred_pos[batch_i].min(), val_plot_pred_pos[batch_i].max())

		# axs[1].plot(val_plot_pred_vel[batch_i], ls='--', color=color, alpha=0.5)
		# axs[1].plot(val_plot_vel[batch_i], color=color, alpha=0.5)

		axs[1].plot(train_plot_pred_pos[batch_i, :, 0], train_plot_pred_pos[batch_i, :, 1], ls='--', color=color, alpha=0.5)
		axs[1].plot(train_plot_pos[batch_i, :, 0], train_plot_pos[batch_i, :, 1], color=color, alpha=0.5)

	# axs[3].plot(train_plot_pred_vel[batch_i], ls='--', color=color, alpha=0.5)
	# axs[3].plot(train_plot_vel[batch_i], color=color, alpha=0.5)

	# axs[2].set_ylim(train_plot_pos.min() * 3, train_plot_pos.max() * 3)
	# axs[3].set_ylim(train_plot_vel.min() * 3, train_plot_vel.max() * 3)


# for traj in self.trajectories:

# plt.plot(traj[:, 0], traj[:, 1], color=traj_color)
# plt.scatter(traj[0, 0], traj[0, 1], marker='*', color=traj_color)

def plot_Lorenz_data(hparams, batch_pred, batch_y, batch_val_pred, batch_val_y):
	val_plot_pred_pos = batch_val_pred[:6, :, :3].cpu().detach()
	val_plot_pos = batch_val_y[:6, :, :3].cpu().detach()

	train_plot_pred_pos = batch_pred[:6, :, :3].cpu().detach()
	train_plot_pos = batch_y[:6, :, :3].cpu().detach()

	fig, axs = plt.subplots(1, 2, sharex=True)
	plt.suptitle(hparams.logname + ' ' + hparams.model)
	axs = axs.flatten()

	cmap = plt.cm.get_cmap('viridis')
	colors = cmap(np.linspace(0, 1, val_plot_pos.shape[0]))

	for batch_i, color in enumerate(colors):
		# print(train_plot_pred_pos.shape, val_plot_pred_pos.shape, batch_i)
		# exit()
		axs[0].plot(val_plot_pred_pos[batch_i, :, 0], val_plot_pred_pos[batch_i, :, 1], ls='--', color=color, alpha=0.5, label='pred')
		axs[0].plot(val_plot_pos[batch_i, :, 0], val_plot_pos[batch_i, :, 1], color=color, alpha=0.5)
		axs[0].legend()
		# axs[0].set_ylim(val_plot_pred_pos[batch_i].min(), val_plot_pred_pos[batch_i].max())

		# axs[1].plot(val_plot_pred_vel[batch_i], ls='--', color=color, alpha=0.5)
		# axs[1].plot(val_plot_vel[batch_i], color=color, alpha=0.5)

		axs[1].plot(train_plot_pred_pos[batch_i, :, 0], train_plot_pred_pos[batch_i, :, 1], ls='--', color=color, alpha=0.5)
		axs[1].plot(train_plot_pos[batch_i, :, 0], train_plot_pos[batch_i, :, 1], color=color, alpha=0.5)

# axs[3].plot(train_plot_pred_vel[batch_i], ls='--', color=color, alpha=0.5)
# axs[3].plot(train_plot_vel[batch_i], color=color, alpha=0.5)

# axs[2].set_ylim(train_plot_pos.min() * 3, train_plot_pos.max() * 3)
# axs[3].set_ylim(train_plot_vel.min() * 3, train_plot_vel.max() * 3)

# for traj in self.trajectories:

# plt.plot(traj[:, 0], traj[:, 1], color=traj_color)
# plt.scatter(traj[0, 0], traj[0, 1], marker='*', color=traj_color)
def npz_to_xyz(data):
	assert data.dim() == data.shape[2] == 3

	frames = []

	for pos in data:
		frames += [Atoms(positions=pos)]

	return frames