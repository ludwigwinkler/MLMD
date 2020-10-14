import os, shutil
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
from DiffEqNets.DiffEqNets_DataUtils import TimeSeries_DataSet, BiDirectional_TimeSeries_DataSet, Sequential_BiDirectional_TimeSeries_DataSet, Sequential_TimeSeries_DataSet

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class SinCos_DataModule(LightningDataModule):

	def __init__(self):

		self.prepare_data()

	def prepare_data(self, *args, **kwargs):

		t = torch.linspace(0,10,50).reshape(1,-1,1)

		x = torch.cat([t.cos(), t.sin()], dim=-1)
		v = (x[:,1:] - x[:,:-1])

		self.data = torch.cat([x[:,:-1,:],v], dim=-1)

		assert self.data.dim()==3
		assert self.data.shape[-1]==4

	def setup(self):
		pass


class QuantumMachine_DFT(LightningDataModule):
	'''
	Benzene: 49862, 72
	'''

	def __init__(self, hparams):

		self.data_path = '../data/'
		self.data_str = hparams.data_set
		self.hparams = hparams

		self.sequential_sampling=False

	def prepare_data(self, *args, **kwargs):

		path_npz = "../data/" + self.data_str
		# path_pt = "../data/" + self.data_str.split(".")[0] + ".pt"

		url = "http://www.quantum-machine.org/gdml/data/npz/" + self.data_str

		if not os.path.exists('../data/' + self.data_str):
			print(f'Downloading {self.data_str} from quantum-machine.org/gdml/data/npz')
			urllib.request.urlretrieve(url, '../data/' + self.data_str)

		if not os.path.exists("../data/" + self.data_str.split(".")[0] + ".pt"):
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

			# plt.hist(pos.flatten(), bins=100, density=True)
			# plt.hist(vel.flatten(), bins=100, density=True)
			# plt.show()
			# exit()

			self.data = torch.cat([pos, vel], dim=-1)
			# torch.save(data, path_pt)

	def setup(self, stage: Optional[str] = None):

		path_pt = "../data/" + self.data_str.split(".")[0] + ".pt"

		data = self.data.clone() # copy data such that the original data is not changed when working with data
		assert data.dim() == 3
		assert data.shape[0] == 1

		'''
		data.shape = [ts, timesteps, features]
		'''
		self.data_mean = data.mean(dim=[0,1])
		self.data_std = data.std(dim=[0,1])

		data = (data - self.data_mean) / (self.data_std + 1e-8)

		train_data_size = int(data.shape[1] * self.hparams.val_split * self.hparams.pct_data_set)
		val_data_size = int(data.shape[1]*(1-self.hparams.val_split))

		'''
		val_split: point in timeseries from where we consider it the validation trajectory
		'''
		val_split = int(data.shape[1] * self.hparams.val_split)
		data_train = data[:, :int(val_split*self.hparams.pct_data_set)] # future reduce val_split by training percentage
		data_val = data[:, int(val_split * self.hparams.pct_data_set):int(val_split * self.hparams.pct_data_set+val_data_size)]

		self.y_mu 	= data_train.data.mean(dim=[0, 1]).to(device)
		self.y_std 	= data_train.data.std(dim=[0, 1]).to(device)

		self.dy 	= (data_train.data[:, 2:, :] - data_train.data[:, :-2, :]) / 2
		self.dy_mu 	= self.dy.mean(dim=[0, 1]).unsqueeze(0).to(device)  # shape = [bs, f]
		self.dy_std 	= self.dy.std(dim=[0, 1]).unsqueeze(0).to(device)  # shape = [bs, f]

		if 'bi' in self.hparams.model:
			self.data_train = BiDirectional_TimeSeries_DataSet(data=data_train,
									   input_length=self.hparams.input_length,
									   output_length=self.hparams.output_length,
									   traj_repetition=self.hparams.train_traj_repetition,
									   sample_axis='timesteps')
			self.data_val = BiDirectional_TimeSeries_DataSet(data=data_val,
									 input_length=self.hparams.input_length,
									 output_length=self.hparams.output_length,
									 traj_repetition=self.hparams.train_traj_repetition,
									 sample_axis='timesteps')

		else:  # the unidirectional case
			self.data_train = TimeSeries_DataSet(data=data_train,
							     input_length=self.hparams.input_length,
							     output_length=self.hparams.output_length,
							     traj_repetition=self.hparams.train_traj_repetition)
			self.data_val = TimeSeries_DataSet(data=data_val,
							   input_length=self.hparams.input_length,
							   output_length=self.hparams.output_length,
							   traj_repetition=self.hparams.train_traj_repetition)

	def train_dataloader(self, *args, **kwargs) -> DataLoader:

		dataloader = DataLoader(self.data_train, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

		return dataloader

	def val_dataloader(self, *args, **kwargs) -> DataLoader:
		return DataLoader(self.data_val, batch_size=self.hparams.batch_size*2, num_workers=self.hparams.num_workers)

	def __repr__(self):

		return f'{self.data_str}: {self.data.shape} features'

class Keto_DFT(LightningDataModule):

	def __init__(self, hparams):

		self.data_path = '../data/'
		self.data_str = hparams.data_set
		self.hparams = hparams

		self.sequential_sampling=False

	def prepare_data(self, *args, **kwargs):

		path_npz = "../data/" + self.data_str
		path_pt = "../data/" + self.data_str.split(".")[0] + ".pt"

		if self.data_str=='keto_1fs.npz':

			pos = np.load('../data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.300K.CLMD.1B_DT-1.0FS_01.POSITION.0.npz')['R']
			vel = np.load('../data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.300K.CLMD.1B_DT-1.0FS_01.VELOCITIES.0.npz')['V']

		if self.data_str=='keto_0.2fs.npz':

			pos = np.load('../data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.300K.CLMD.1B_DT-0.2FS_01.POSITION.0.npz')['R']
			vel = np.load('../data/KETO-MDA.SGDML.CCSD-T.CC-PVDZ.300K.CLMD.1B_DT-0.2FS_01.VELOCITIES.0.npz')['V']

		# print([key for key in pos.keys()])
		# print([key for key in vel.keys()])

		pos = torch.from_numpy(pos).float()
		vel = torch.from_numpy(vel).float()

		pos = pos.flatten(-2, -1)
		vel = vel.flatten(-2, -1)

		self.data = torch.cat([pos, vel], dim=-1)

	def setup(self, stage: Optional[str] = None):

		path_pt = "../data/" + self.data_str.split(".")[0] + ".pt"

		# data = torch.load(path_pt).float()
		data = self.data
		assert data.dim() == 3
		assert data.shape[0] == 1

		# plt.hist(data[:,:,data.shape[-1]//2], bins=100, density=True)
		# plt.show()
		self.data_mean = data.mean(dim=[0,1])
		self.data_std = data.std(dim=[0,1])

		data = (data - self.data_mean) / (self.data_std + 1e-8)

		train_data_size = int(data.shape[1] * self.hparams.val_split * self.hparams.pct_data_set)
		val_data_size = int(data.shape[1] - data.shape[1]*self.hparams.val_split)

		val_split = int(data.shape[1] * self.hparams.val_split)
		data_train = data[:, :int(val_split*self.hparams.pct_data_set)]
		data_val = data[:, int(val_split * self.hparams.pct_data_set):int(val_split * self.hparams.pct_data_set+val_data_size)]

		self.y_mu 	= data_train.data.mean(dim=[0, 1]).to(device)
		self.y_std 	= data_train.data.std(dim=[0, 1]).to(device)

		self.dy 	= (data_train.data[:, 2:, :] - data_train.data[:, :-2, :]) / 2
		self.dy_mu 	= self.dy.mean(dim=[0, 1]).unsqueeze(0).to(device)  # shape = [bs, f]
		self.dy_std 	= self.dy.std(dim=[0, 1]).unsqueeze(0).to(device)  # shape = [bs, f]

		if 'bi' in self.hparams.model:
			self.data_train = BiDirectional_TimeSeries_DataSet(data=data_train,
									   input_length=self.hparams.input_length,
									   output_length=self.hparams.output_length,
									   traj_repetition=self.hparams.train_traj_repetition,
									   sample_axis='timesteps')
			self.data_val = BiDirectional_TimeSeries_DataSet(data=data_val,
									 input_length=self.hparams.input_length,
									 output_length=self.hparams.output_length,
									 traj_repetition=self.hparams.train_traj_repetition,
									 sample_axis='timesteps')

		else:  # the unidirectional case
			self.data_train = TimeSeries_DataSet(data=data_train,
							     input_length=self.hparams.input_length,
							     output_length=self.hparams.output_length,
							     traj_repetition=self.hparams.train_traj_repetition)
			self.data_val = TimeSeries_DataSet(data=data_val,
							   input_length=self.hparams.input_length,
							   output_length=self.hparams.output_length,
							   traj_repetition=self.hparams.train_traj_repetition)

	def train_dataloader(self, *args, **kwargs) -> DataLoader:

		dataloader = DataLoader(self.data_train, batch_size=self.hparams.batch_size, shuffle=True, num_workers=self.hparams.num_workers)

		return dataloader

	def val_dataloader(self, *args, **kwargs) -> DataLoader:
		return DataLoader(self.data_val, batch_size=self.hparams.batch_size*2, num_workers=self.hparams.num_workers)

	def __len__(self):
		self.prepare_data()
		self.setup()
		return self.data.shape[1]

	@property
	def shape(self):
		return self.data.shape

	def __repr__(self):
		return f'{self.data_str}: {self.data.shape} features'

class HMC_DM(LightningDataModule):

	def __init__(self, hparams):

		self.hparams = hparams

	def setup(self, *args, **kwargs):

		from DiffEqNets.MolecularDynamics.data.HMC_Data_Generation import HMCData

		num_datasets = 1
		num_trajectories = 2000
		num_means = 10
		trajectory_stepsize = 0.2
		dist_mean_min_max = 4
		num_steps = 100

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

		data = torch.cat(trajectories, dim=0).squeeze(0)
		surfaces = torch.cat(surfaces, dim=0)
		means = torch.cat(means, dim=0)
		covars = torch.cat(covars, dim=0)

		assert data.dim() == 3

		data = (data - data.mean(dim=[0, 1])) / (data.std(dim=[0, 1]) + 1e-3)

		val_split = int(data.shape[1] * self.hparams.val_split)
		data_train, data_val = data[:, :val_split], data[:, val_split:]

		self.y_mu = data_train.data.mean(dim=[0, 1]).to(device)
		self.y_std = data_train.data.std(dim=[0, 1]).to(device)

		self.dy = (data_train.data[:, 2:, :] - data_train.data[:, :-2, :]) / 2
		self.dy_mu = self.dy.mean(dim=[0, 1]).unsqueeze(0).to(device)  # shape = [bs, f]
		self.dy_std = self.dy.std(dim=[0, 1]).unsqueeze(0).to(device)  # shape = [bs, f]

		if 'bi' in self.hparams.model:
			self.data_train = BiDirectional_TimeSeries_DataSet(data=data_train,
									   input_length=self.hparams.input_length,
									   output_length=self.hparams.output_length,
									   traj_repetition=self.hparams.train_traj_repetition,
									   sample_axis='trajs')
			self.data_val = BiDirectional_TimeSeries_DataSet(data=data_val,
									 input_length=self.hparams.input_length,
									 output_length=self.hparams.output_length,
									 traj_repetition=self.hparams.train_traj_repetition,
									 sample_axis='trajs')

		else:  # the unidirectional case
			self.data_train = TimeSeries_DataSet(data=data_train,
							     input_length=self.hparams.input_length,
							     output_length=self.hparams.output_length,
							     traj_repetition=self.hparams.train_traj_repetition)
			self.data_val = TimeSeries_DataSet(data=data_val,
							   input_length=self.hparams.input_length,
							   output_length=self.hparams.output_length,
							   traj_repetition=self.hparams.train_traj_repetition)

	def train_dataloader(self, *args, **kwargs) -> DataLoader:
		# print(f" @train_dataloader(): {self.data_train.data.shape=}")
		return DataLoader(self.data_train, batch_size=self.hparams.batch_size, shuffle=True)

	def val_dataloader(self, *args, **kwargs) -> DataLoader:
		return DataLoader(self.data_val, batch_size=self.hparams.batch_size, shuffle=True)

def load_dm_data(hparams):
	data_str = hparams.data_set
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

	elif data_str in ['keto_1fs.npz', 'keto_0.2fs.npz']:

		dm = Keto_DFT(hparams)

	elif data_str in ['hmc']:
		dm = HMC_DM(hparams)
	elif data_str=='sincos':
		dm = SinCos_DataModule()
	else:
		exit(f"No valid dataset provided ...")

	dm.prepare_data()
	dm.setup()

	y_mu = dm.data_train.data.mean(dim=[0, 1])
	y_std = dm.data_train.data.std(dim=[0, 1])

	dy = (dm.data_train.data[:, 2:, :] - dm.data_train.data[:, :-2, :]) / 2

	dy_mu = dy.mean(dim=[0, 1]).unsqueeze(0)  # shape = [bs, f]
	dy_std = dy.std(dim=[0, 1]).unsqueeze(0)  # shape = [bs, f]

	return dm


def load_data(hparams, normalize=True, plot=True, analyze=True):
	if hparams.train_data in ['benzene_dft.npz',
				  'ethanol_dft.npz',
				  'malonaldehyde_dft.npz']:
		assert os.path.exists('../data/' + hparams.train_data), f"{'../data/' + hparams.train_data} doesnt exists"

		data = load_and_process_QuantumMachine_data(plot=plot, normalize=normalize, analyze=analyze, string=hparams.train_data)
		val_split = int(data.shape[1] * hparams.val_split)  # hparams.val_split batch_i.e. 0.7, 0.8
		train_data = data[:, :val_split, :]
		val_data = data[:, val_split:, :]

	elif hparams.train_data in ['H2O/HigherEnergy/H2O_HigherEnergy1.npz',
				    'H2O/HigherEnergy/H2O_HigherEnergy2.npz',
				    'H2O/HigherEnergy/H2O_HigherEnergy3.npz',
				    'H2O/LowerEnergy/H2O_LowerEnergy1.npz']:

		'''
		H2 data is one dimensional batch_i.e. oscilates along a single axis
		H20 data is fully three dimensional
		'''
		assert os.path.exists('../data/' + hparams.train_data), f"{'../data/' + hparams.train_data} doesnt exists"

		data = load_and_process_MD_data(plot=plot, analyze=False, normalize=normalize, string=hparams.train_data)
		val_split = int(data.shape[1] * hparams.val_split)  # hparams.val_split batch_i.e. 0.7, 0.8
		train_data = data[:, :val_split, :]
		val_data = data[:, val_split:, :]

	elif hparams.train_data in ['Ethanol']:
		assert os.path.exists('../data/' + hparams.train_data), f"{'../data/' + hparams.train_data} doesnt exists"

		data = load_and_process_Ethanol_data(hparams=hparams, plot=plot)
		val_split = int(data.shape[1] * hparams.val_split)  # hparams.val_split batch_i.e. 0.7, 0.8
		train_data = data[:, :val_split, :]
		val_data = data[:, val_split:, :]

	elif hparams.train_data in ['hmc']:

		data = load_and_process_HMC_data(hparams=hparams, plot=plot)
		val_split = int(data.shape[0] * hparams.val_split)  # hparams.val_split batch_i.e. 0.7, 0.8
		train_data = data[:val_split, :, :]
		val_data = data[val_split:, :, :]

	elif hparams.train_data in ['lorenz']:

		data = load_and_process_Lorenz_data(hparams=hparams, plot=True)
		# print(f"{data.shape=}")
		# exit()
		val_split = int(data.shape[1] * hparams.val_split)  # hparams.val_split batch_i.e. 0.7, 0.8
		train_data = data[:, :val_split, :]
		val_data = data[:, val_split:, :]

	else:
		raise ValueError(f'Data not found: {hparams.train_data}')

	if hparams.subsampling > 1:
		data = data[:, ::int(hparams.subsampling), :]

	if hparams.num_samples > 1:
		train_data = train_data[:, :hparams.num_samples, :]
		val_data = val_data[:, -hparams.num_samples // 2:, :]

	assert train_data.dim() == 3

	y_mu = train_data.mean(dim=[0, 1]).unsqueeze(0)
	y_std = train_data.std(dim=[0, 1]).unsqueeze(0)

	dy = (train_data[:, 2:, :] - train_data[:, :-2, :]) / 2

	dy_mu = dy.mean(dim=[0, 1]).unsqueeze(0)  # shape = [bs, f]
	dy_std = dy.std(dim=[0, 1]).unsqueeze(0)  # shape = [bs, f]

	return train_data, val_data, y_mu, y_std, dy_mu, dy_std


def generate_toy_trajectory(_num_samples=100, _time_length=2 * np.pi, _plot=True):
	t = torch.linspace(0, _time_length, _num_samples)

	x = torch.sin(t)
	dx = torch.cos(t)

	data = torch.cat([x.unsqueeze(-1), dx.unsqueeze(-1)], dim=1).unsqueeze(0)

	if _plot:
		plt.plot(t, x, label='x')
		plt.plot(t, dx, label='dx')
		plt.grid()
		plt.legend()
		plt.show()

	return data


def load_and_process_HMC_data(hparams, normalize=True, plot=False, analyze=False):
	from DiffEqNets.MolecularDynamics.data.HMC_Data_Generation import HMCData

	num_datasets = 1
	num_trajectories = 2000
	num_means = 10
	trajectory_stepsize = 0.2
	dist_mean_min_max = 4
	num_steps = 100

	means, covars, surfaces, trajectories = [], [], [], []
	for i_dataset in range(num_datasets):
		data_gen = HMCData(num_trajs=num_trajectories, min_max=dist_mean_min_max, num_steps=num_steps,
				   num_means=num_means, step_size=trajectory_stepsize)
		surface, X_grid, Y_grid = data_gen.generate_plottingsurface()
		surface = (surface - surface.min()) / (surface.max() - surface.min())
		trajs = data_gen.generate_trajectory()
		if plot:
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

	trajectories = torch.cat(trajectories, dim=0).squeeze(0)
	surfaces = torch.cat(surfaces, dim=0)
	means = torch.cat(means, dim=0)
	covars = torch.cat(covars, dim=0)

	# print(f"{trajectories.shape=}")

	return trajectories


def load_and_process_Lorenz_data(hparams, normalize=True, plot=True, analyze=False):
	from DiffEqNets.MolecularDynamics.data.Lorenz_Data_Generation import LorenzData
	data_gen = LorenzData()

	trajectories = data_gen.generate_trajectory()

	if normalize:
		trajectories = (trajectories - trajectories.mean(dim=[0, 1])) / (trajectories.std(dim=[0, 1]) + 1e-3)
	# exit()

	return trajectories


def load_and_process_Ethanol_data(hparams, normalize=True, plot=True, analyze=False):
	pos = torch.from_numpy(np.load('../data/Ethanol/ETHANOL.300K.1B.SYMMK.NVE.CCSD-T.01.POSITION.0.npz')[
				       'R']).cpu().float()  # shape = [1, timesteps, atoms, dims]
	vel = torch.from_numpy(np.load('../data/Ethanol/ETHANOL.300K.1B.SYMMK.NVE.CCSD-T.01.VELOCITIES.0.npz')[
				       'V']).cpu().float()  # shape = [1, timesteps, atoms, dims]

	if normalize:
		center_of_mass = pos.mean(dim=2, keepdim=True)
		pos -= center_of_mass

	if analyze:
		vel_samples = torch.sum(vel ** 2, dim=-1) ** 0.5
		plt.hist(vel_samples[:, :, 0])

	pos = pos.flatten(start_dim=-2, end_dim=-1)
	vel = vel.flatten(start_dim=-2, end_dim=-1)

	if hparams.subsampling > 1:
		pos = pos[:, ::int(hparams.subsampling), :]
		vel = (pos[:, 2:, :] - pos[:, :-2, :]) / 2
		pos = pos[:, 1:-1, :]
	# print(f'{vel.shape=} {pos.shape=}')

	# vel = vel.flatten(start_dim=-2, end_dim=-1)
	# vel = (pos[:,2:,:] - pos[:,:-2])/2
	# pos = pos[:,1:-1,:]

	data = torch.cat([pos, vel], dim=-1).float()  # shape = [ timeseries, timesteps, [pos, vel]^T ]

	# if _normalize:
	# 	data = (data - data.mean(dim=[0,1]))/(data.std(dim=[0,1])+1e-2)

	if plot:

		for T in [100, 1000, 10000]:
			t0 = torch.randint(0, data.shape[1] - T, (1,))

			plot_data = data[0, t0:t0 + T, :]
			fig, axs = plt.subplots(2, 1)
			axs = axs.flatten()

			axs[0].plot(pos[0, t0:t0 + T, :3], alpha=0.5)
			axs[1].plot(vel[0, t0:t0 + T, :3], alpha=0.5)
			axs[0].set_title('pos')
			axs[1].set_title('vel')

			# axs[:].legend()
			plt.show()

	return data


def load_and_process_H2_data(_normalize=True, _plot=True, _analyze=True):
	raw_data = np.load('H2.npz')
	pos = torch.from_numpy(raw_data['R']).cpu().float().unsqueeze(0)
	vel = torch.from_numpy(raw_data['V']).cpu().float().unsqueeze(0)

	print(pos[0, :10, 0, :])
	print(pos[0, :10, 1, :])
	# print(pos.shape, vel.shape)
	# exit()

	# pos = torch.from_numpy(np.load('ETHANOL.300K.1B.SYMMK.NVE.CCSD-T.01.POSITION.0.npz')['R']).cpu().float() # shape = [1, timesteps, atoms, dims]
	# vel = torch.from_numpy(np.load('ETHANOL.300K.1B.SYMMK.NVE.CCSD-T.01.VELOCITIES.0.npz')['V']).cpu().float() # shape = [1, timesteps, atoms, dims]

	if _normalize:
		center_of_mass = pos.mean(dim=2, keepdim=True)
		pos -= center_of_mass

	if _analyze:
		vel_samples = torch.sum(vel ** 2, dim=-1) ** 0.5

		plt.hist(vel_samples[:, :, 0])

	pos = pos.flatten(start_dim=-2, end_dim=-1)
	vel = vel.flatten(start_dim=-2, end_dim=-1)

	data = torch.cat([pos, vel], dim=-1).float()  # shape = [ timeseries, timesteps, [pos, vel]^T ]

	if _normalize:
		data = (data - data.mean(dim=[0, 1])) / (data.std(dim=[0, 1]) + 1e-2)

	if _plot:
		if True:
			T = 1000
			t0 = torch.randint(0, data.shape[1] - T, (1,))
		else:
			T = data.shape[1]
			t0 = 0

		plot_data = data[0, t0:t0 + T, :]
		fig, axs = plt.subplots(2, 1)
		axs = axs.flatten()

		axs[0].plot(plot_data[:, :], alpha=0.5)
		axs[1].plot(plot_data[:, :], alpha=0.5)
		axs[0].set_title('pos')
		axs[1].set_title('vel')

		# axs[:].legend()
		plt.show()

	return data


def load_and_process_MD_data(normalize=True, plot=True, analyze=True, string=None):
	string = "../data/" + string

	raw_data = np.load(string)
	pos = torch.from_numpy(raw_data['R']).cpu().float().unsqueeze(0)
	vel = torch.from_numpy(raw_data['V']).cpu().float().unsqueeze(0)

	# pos = torch.from_numpy(np.load('ETHANOL.300K.1B.SYMMK.NVE.CCSD-T.01.POSITION.0.npz')['R']).cpu().float() # shape = [1, timesteps, atoms, dims]
	# vel = torch.from_numpy(np.load('ETHANOL.300K.1B.SYMMK.NVE.CCSD-T.01.VELOCITIES.0.npz')['V']).cpu().float() # shape = [1, timesteps, atoms, dims]

	if normalize:
		center_of_mass = pos.mean(dim=2, keepdim=True)
		pos -= center_of_mass

	if analyze:
		vel_samples = torch.sum(vel ** 2, dim=-1) ** 0.5

		plt.hist(vel_samples[:, :, 0])

	pos = pos.flatten(-2, -1)
	vel = vel.flatten(-2, -1)

	# print(torch.sum(pos - pos_.reshape(*pos.shape)))

	data = torch.cat([pos, vel], dim=-1).float()  # shape = [ timeseries, timesteps, [pos, vel] ]

	if normalize:
		data = (data - data.mean(dim=[0, 1])) / (data.std(dim=[0, 1]) + 1e-2)

	if plot:

		# from scipy.fftpack import fft, fftshift
		# NFFT=1024

		# plt.show()

		# exit()
		for T in [100, 1000, 10000]:
			t0 = torch.randint(0, data.shape[1] - T, (1,))

			# plot_data = data[0, t0:t0 + T, :]
			fig, axs = plt.subplots(2, 1)
			axs = axs.flatten()

			axs[0].plot(pos[0, t0:t0 + T, :3], alpha=0.5)
			axs[1].plot(vel[0, t0:t0 + T, :3], alpha=0.5)
			axs[0].set_title('pos')
			axs[1].set_title('vel')

			# axs[:].legend()
			plt.show()
		exit()

	# pos_, vel_ = torch.chunk(data, chunks=2, dim=-1)

	# pos_ = pos_.reshape(*pos.shape)

	# print(torch.sum(pos - pos_.reshape(*pos.shape)))
	# exit()

	return data


def load_and_process_QuantumMachine_data(normalize=True, plot=True, analyze=True, string=None):
	'''
	loads data from quantum-machine.org's npz folder
	for example: http://www.quantum-machine.org/gdml/data/npz/naphthalene_dft.npz
	'''

	assert string.split(".")[1] == 'npz'
	path_npz = "../data/" + string
	path_pt = "../data/" + string.split(".")[0] + ".pt"

	url = "http://www.quantum-machine.org/gdml/data/npz/" + string
	if not os.path.exists('../data/' + string):
		print(f'Downloading {string} from quantum-machine.org/gdml/data/npz')
		urllib.request.urlretrieve(url, '../data/' + string)

	if os.path.exists("/data/" + string.split(".")[0] + ".pt"):
		print("Processed data exists")
		data = torch.load(path_pt)
	else:
		data = np.load(path_npz)
		try:
			data = data['R']
		except:
			print("Loaded data doesnt have a 'R' key ")

		data = torch.from_numpy(data).double()

		pos = data[:-1]
		# vel = (data[2:] - data[:-2])/2
		vel = (data[1:] - data[:-1])

		assert torch.isclose((data[1:] - (pos + vel)).sum(), torch.scalar_tensor(0.0, dtype=torch.double))

		# print(f'{data.shape=} {vel.shape=}')
		# print(f'{(data[1:] - (pos+vel)).sum()=}')

		# pos = torch.from_numpy(pos)
		# vel = torch.from_numpy(vel)

		if analyze:
			print(f'{data.shape=}')
			print(f'{pos.mean(dim=[0,1])=}, {pos.std(dim=[0,1])=}')
			print(f'{vel.mean(dim=[0,1])=}, {vel.std(dim=[0,1])=}')

			plt.plot()

		pos = pos.flatten(-2, -1)
		vel = vel.flatten(-2, -1)

		data = torch.cat([pos, vel], dim=-1)
		torch.save(data, path_pt)

	if plot:

		pos, vel = torch.chunk(data, chunks=2, dim=-1)
		num_atoms = pos.shape[-1] // 3
		pos = pos.reshape(data.shape[0], num_atoms, 3)
		vel = vel.reshape(data.shape[0], num_atoms, 3)
		fig, axs = plt.subplots(1, 2, sharex=True)
		fig.suptitle(string)
		axs = axs.flatten()
		print(f'{num_atoms=}')
		for atom_i in range(num_atoms):
			for dim in range(3):
				axs[0].plot(pos[:1000, atom_i, dim])
				axs[1].plot(vel[:1000, atom_i, dim])

		axs[0].set_title('Position')
		axs[1].set_title('Velocity')
		plt.show()

	if normalize:
		data = (data - data.mean(dim=0)) / (data.std(dim=0))

	smooth = False
	if smooth:
		# print(data.shape)
		plt.plot(data[:1000, :10], label='data')
		N = 51
		filter = torch.ones(1, N, dtype=torch.double) / N
		filter = filter.unsqueeze(1)
		data = data.unsqueeze(1)
		data = F.conv1d(data.T, weight=filter, padding=N // 2).squeeze().T
	# print(data.shape)
	# plt.plot(data[:1000,:10], label='smoothed')
	# plt.legend()
	# plt.show()
	# exit('smooth @load QM data')

	if analyze or plot:
		exit(f'@load_and_process_QuantumMachine_data')
		pass

	data = data.unsqueeze(0).float()

	return data


def compute_ase_angle_distance(data, atom_type_str='H20'):
	assert data.ndim == 3, f'data should be shape=[BS, t, F] but is {data.shape}'

	pos, vel = torch.chunk(data, chunks=2, dim=-1)  # shape = [BS, t, Atoms*cartesian dims] for both pos & vel
	pos = pos.reshape(pos.shape[0], pos.shape[1], pos.shape[2] // 3, 3)  # shape = [BS, t, Atoms, cartesian dims]
	vel = vel.reshape(vel.shape[0], vel.shape[1], vel.shape[2] // 3, 3)  # shape = [BS, t, Atoms, cartesian dims]

	pos = pos.flatten(0, 1).cpu()
	vel = vel.flatten(0, 1).cpu()

	u = pos[:, 1, :] - pos[:, 0, :]
	v = pos[:, 1, :] - pos[:, 2, :]

	torch_angle = torch.sum(u * v, dim=-1) / (u.norm(dim=-1) * v.norm(dim=-1) + 1e-3)  # ∈ (-1, 1)

	# print(f'{torch_angle=}')
	torch_degree = torch_angle
	# torch_angle = torch.acos(torch_angle.clamp(-1+1e-3,1-1e-3)) # ∈ (-2*π, 2*π)
	# torch_degree = torch_angle * 360/(2*math.pi) # ∈ (-180 degree, 180 degree) to check with ASE environment

	dist = pos.unsqueeze(1) - pos.unsqueeze(2)
	dist = dist ** 2
	torch_dist = dist.sum(dim=-1) ** 0.5

	return torch_degree, torch_dist


def ase_loss(pred, target, atom_type_str='H20'):
	pred_degree, pred_dist = compute_ase_angle_distance(pred, atom_type_str)
	target_degree, target_dist = compute_ase_angle_distance(target, atom_type_str)

	angle_loss = F.mse_loss(pred_degree, target_degree)
	dist_loss = F.mse_loss(pred_dist, target_dist)

	# print(f'{angle_loss=}')
	# print(f'{dist_loss=}')

	loss = angle_loss + dist_loss

	return loss


def compute_angles_and_innerdistance(data, atom_type_str='H2O'):
	import sys
	np.set_printoptions(threshold=sys.maxsize)

	assert data.ndim == 3, f'data should be shape=[BS, t, F] but is {data.shape}'

	pos, vel = torch.chunk(data, chunks=2, dim=-1)  # shape = [BS, t, Atoms*cartesian dims] for both pos & vel
	pos = pos.reshape(pos.shape[0], pos.shape[1], pos.shape[2] // 3, 3)  # shape = [BS, t, Atoms, cartesian dims]
	vel = vel.reshape(vel.shape[0], vel.shape[1], vel.shape[2] // 3, 3)  # shape = [BS, t, Atoms, cartesian dims]

	pos = pos.flatten(0, 1).cpu()
	vel = vel.flatten(0, 1).cpu()

	distances = []
	angles = []

	u = pos[:, 1, :] - pos[:, 0, :]
	v = pos[:, 1, :] - pos[:, 2, :]

	# print(u.shape, v.shape);exit()
	nominator = torch.sum(u * v, dim=-1)
	denominator = (u.norm(dim=-1) * v.norm(dim=-1))
	# print(f'{denominator=}')
	torch_angle = nominator / denominator  # ∈ (-1, 1)

	if torch.isnan(torch_angle).any():
		print('Nan detected:')
		print(f'{torch_angle.T=}')
		exit()

	torch_angle = torch.acos(torch_angle.clamp(-1, 1))  # ∈ (-2*π, 2*π)
	torch_degree = torch_angle * 360 / (2 * math.pi)  # ∈ (-180 degree, 180 degree) to check with ASE environment

	# print(f'{torch_degree=}')

	if torch.isnan(torch_degree).any():
		print('Nan detected:')
		print(f'{torch_degree.T=}')
		exit()

	if (torch_angle.max().abs() >= 2 * math.pi).any() or (torch_angle.min().abs() >= 1.0).any():
		raise ValueError(f'{torch_angle.max()=}')

	np_angle = torch_angle.cpu().numpy()
	if np.isnan(np_angle).any():
		print('Nan detected:')
		print(np_angle.T)
		exit()

	np_angle = np.nan_to_num(np_angle)

	# print(f'{np_angle[0]=}')

	np_degrees = np.degrees(np_angle)
	# print(f'{np_degrees[0]=}')
	# exit()
	# if torch.isnan(torch_angles).sum()>0:
	# 	print(torch_angles[torch.isnan(torch_angles)])
	# print(torch_angles[torch.isnan(torch_angles)])

	torch_angles = torch.from_numpy(np_degrees).unsqueeze(-1)

	# print(torch_angles.shape)

	dist = pos.unsqueeze(1) - pos.unsqueeze(2)
	dist = dist ** 2
	torch_dists = dist.sum(dim=-1) ** 0.5

	assert torch_angles.dim() == 2
	assert torch_dists.dim() == 3
	assert torch_dists.shape[1] == torch_dists.shape[2]

	atom = Atoms(atom_type_str, positions=pos[0].tolist(), velocities=vel[0].tolist())
	ase_angle = atom.get_angles(indices=[3 * [0, 1, 2]])
	ase_distance = atom.get_all_distances()

	if torch.sum((torch_degree[0] - ase_angle) ** 2) > 0.1 or torch.sum((torch_dists[0] - ase_distance) ** 2) > 1e-2:
		print('torch_dist')
		print(torch_dists[0])
		print('ase dist')
		print(ase_distance)

		print('torch_angle')
		print(torch_angles[0])
		print('ase angle')
		print(ase_angle)

		print(f'{torch.sum((torch_angles[0] - ase_angle)**2)=}')
		print(f'{torch.sum((torch_dists[0] - ase_distance)**2)=}')
		raise ValueError('Molecular angle and distance computation deviates more than 1e-2 from the ASE computation')
		exit()

	if False:
		# in Order to check computations with ASE framework
		for i, (pos_t, vel_t) in enumerate(zip(pos, vel)):
			# print(pos_t.shape, vel_t.shape)
			# pos_t=pos_t.T.contiguous()
			# vel_t=vel_t.T.contiguous()
			# print(pos_t.shape, vel_t.shape)

			atom = Atoms(atom_type_str, positions=pos_t.tolist(), velocities=vel_t.tolist())
			angle = atom.get_angles(indices=[3 * [0, 1, 2]])
			# print('In compute angles and inner distances')
			# print(angle.shape)
			# print('OUt compute angles and inner distances')
			distance = atom.get_all_distances()
			print('dist', distance)

			# exit()

			# print(pos_t.shape)
			u = pos_t[1] - pos_t[0]
			v = pos_t[1] - pos_t[2]

			# print(u, v)

			np_angle = torch.acos(torch.sum(u * v) / (u.norm(dim=0) * v.norm(dim=0))).reshape(1, 1).numpy()
			# print('np angle', np_angle)
			torch_angle = torch.from_numpy(np.degrees(np_angle))
			print('torch_angle', torch_angle)
			print('angle', angle)

			exit() if i == 2 else None

		# distances.append(distance)
		# angles.append(angle)

		angles = torch.Tensor(angles)
		distances = torch.Tensor(distances)
	return torch_degree, torch_dists


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