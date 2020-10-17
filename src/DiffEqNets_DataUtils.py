import torch
import numpy as np

import matplotlib.pyplot as plt

from torch.utils.data import Dataset, ConcatDataset
import torch.nn.functional as F

def to_np(_tensor):
	if _tensor is not None:
		assert isinstance(_tensor, torch.Tensor)
		return _tensor.cpu().squeeze().detach().numpy()
	else:
		return None

class TimeSeries_DataSet(Dataset):

	def __init__(self, data, input_length=1, output_length=2, traj_repetition=1, sample_axis=None):

		assert data.dim()==3, f'Data.dim()={data.dim()} and not [trajs, steps, features]'
		assert type(input_length)==int
		assert type(output_length)==int
		assert input_length>=1
		assert output_length>=1

		if sample_axis is not None:
			assert sample_axis in ['trajs', 'timesteps'], f'Invalid axis ampling'

		self.input_length = input_length
		self.output_length = output_length
		self.output_length_samplerange = [output_length, output_length+1]

		self.data = data
		self.traj_repetition = traj_repetition

		if sample_axis is None:
			if self.data.shape[0]*self.traj_repetition>=self.data.shape[1]: # more trajs*timesteps than timesteps
				self.sample_axis = 'trajs'
				# print(f'Sampling along trajectory axis {self.data.shape} with dataset multiplier {self.traj_repetition} ->{self.data.shape[0]*self.traj_repetition}')
			elif self.data.shape[0]*self.traj_repetition<self.data.shape[1]: # more timesteps than trajs
				self.sample_axis = 'timesteps'
				# print(f'Sampling along timestep axis {self.data.shape}->{self.data.shape[1]}')
			else:
				raise ValueError('Sample axis not defined in data set')

		elif sample_axis is not None and sample_axis in ['trajs', 'timesteps']:
			self.sample_axis = sample_axis
			if self.sample_axis == 'timesteps':
				# print(f'Sampling along timestep axis {self.data.shape}->{self.data.shape[1]}')
				pass
			if self.sample_axis == 'trajs':
				# print(f'Sampling along trajectory axis {self.data.shape} with dataset multiplier {self.traj_repetition} ->{self.data.shape[0]*self.traj_repetition}')
				pass

	def sample_output_length(self):

		self.output_length = np.random.randint(int(self.output_length_samplerange[0]), int(self.output_length_samplerange[1]))

	def update_output_length_samplerange(self, low=0.1, high=0.5, mode='add'):

		assert mode in ['add', 'set'], 'mode is not set correctly'

		cur_low, cur_high = self.output_length_samplerange[0], self.output_length_samplerange[1]

		if mode=='add':
			if cur_high+high<self.__len__(): cur_high += high
			if cur_low+low < cur_high: cur_low += low
			self.output_length_samplerange = np.array([cur_low, cur_high])
		elif mode=='set' and low < high:
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

		total_length = self.output_length + self.input_length

		if self.sample_axis=='trajs':
			'''
			Many short timeseries
			'''
			idx = idx % self.data.shape[0] # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
			traj = self.data[idx] # selecting trajectory

			assert (traj.shape[0] - total_length) >= 0, f' trajectory length {traj.shape[0]} is smaller than batch_length {self.batchlength}'
			t0 = np.random.choice(traj.shape[0] - total_length) 	# selecting starting time in trajectory

		elif self.sample_axis=='timesteps':
			'''
			Few short timeseries
			'''

			traj_index = np.random.choice(self.data.shape[0]) 	# Randomly select one of the few timeseries
			traj = self.data[traj_index]				# select the timeseries

			t0 = idx						# we're sampling from the timesteps

		y0 = traj[t0:(t0 + self.input_length)]  # selecting corresponding startin gpoint
		target = traj[t0:(t0 + total_length)]  # snippet of trajectory

		if self.input_length==1:
			y0.squeeze_(0)

		return y0, self.output_length, target

	def __len__(self):

		if self.sample_axis=='trajs':
			return self.data.shape[0]*self.traj_repetition
		elif self.sample_axis=='timesteps':
			return self.data.shape[1] - (self.input_length + self.output_length)
		else:
			raise ValueError('Sample axis not defined in data set not defined')

class BiDirectional_TimeSeries_DataSet(Dataset):

	def __init__(self, data, input_length=1, output_length=2, traj_repetition=1, sample_axis=None):

		assert data.dim()==3, f'Data.dim()={data.dim()} and not [trajs, steps, features]'
		assert type(input_length)==int
		assert type(output_length)==int
		assert input_length>=1
		assert output_length>=1

		if sample_axis is not None:
			assert sample_axis in ['trajs', 'timesteps'], f'Invalid axis ampling'

		self.input_length = input_length
		self.output_length = output_length
		self.output_length_samplerange = [output_length, output_length+1]

		self.data = data
		self.traj_repetition = traj_repetition

		if sample_axis is None:
			if self.data.shape[0]*self.traj_repetition>=self.data.shape[1]: # more trajs*timesteps than timesteps
				self.sample_axis = 'trajs'
				# print(f'Sampling along trajectory axis {self.data.shape} with dataset multiplier {self.traj_repetition} ->{self.data.shape[0]*self.traj_repetition}')
			elif self.data.shape[0]*self.traj_repetition<self.data.shape[1]: # more timesteps than trajs
				self.sample_axis = 'timesteps'
				# print(f'Sampling along timestep axis {self.data.shape}->{self.data.shape[1]}')
			else:
				raise ValueError('Sample axis not defined in data set')

		elif sample_axis is not None and sample_axis in ['trajs', 'timesteps']:
			self.sample_axis = sample_axis
			if self.sample_axis == 'timesteps':
				# print(f'Sampling along timestep axis {self.data.shape}->{self.data.shape[1]}')
				pass
			if self.sample_axis == 'trajs':
				# print(f'Sampling along trajectory axis {self.data.shape} with dataset multiplier {self.traj_repetition} ->{self.data.shape[0]*self.traj_repetition}')
				pass

	def sample_output_length(self):

		self.output_length = np.random.randint(int(self.output_length_samplerange[0]), int(self.output_length_samplerange[1]))

	def update_output_length_samplerange(self, low=0.1, high=0.5, mode='add'):

		assert mode in ['add', 'set'], 'mode is not set correctly'

		cur_low, cur_high = self.output_length_samplerange[0], self.output_length_samplerange[1]

		if mode=='add':
			if cur_high+high<self.__len__(): cur_high += high
			if cur_low+low < cur_high: cur_low += low
			self.output_length_samplerange = np.array([cur_low, cur_high])
		elif mode=='set' and low < high:
			assert high < self.__len__()
			self.output_length_samplerange = np.array([low, high])
		else:
			raise ValueError('Incorrect inputs to update_batchlength_samplerange')

	def __getitem__(self, idx):

		total_length = self.input_length + self.output_length + self.input_length

		if self.sample_axis=='trajs':
			'''
			Many short timeseries
			'''
			idx = idx % self.data.shape[0] # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
			traj = self.data[idx] # selecting trajectory

			assert (traj.shape[0] - total_length) >= 0, f' trajectory length {traj.shape[0]} is smaller than batch_length {self.batchlength}'
			t0 = np.random.choice(traj.shape[0] - total_length) 	# selecting starting time in trajectory
			t1 = t0 + self.input_length + self.output_length

		elif self.sample_axis=='timesteps':
			'''
			Few short timeseries
			'''

			traj_index = np.random.choice(self.data.shape[0]) 	# Randomly select one of the few timeseries
			traj = self.data[traj_index]				# select the timeseries

			t0 = idx						# we're sampling from the timesteps
			t1 = t0  + self.input_length + self.output_length
			assert (t0+total_length) < self.data.shape[1]

		'''
		y0 + input_length | output_length | y1 + input_length
		'''

		y0 = traj[t0:(t0 + self.input_length)]  # selecting corresponding starting point
		y1 = traj[t1:(t1 + self.input_length)]  # selecting corresponding starting point

		target = traj[t0:(t0 + total_length)]  # snippet of trajectory


		assert F.mse_loss(y0, target[:self.input_length]) == 0, f'{F.mse_loss(y0, target[0])=}'
		assert F.mse_loss(y1,target[-self.input_length:])==0, f'{F.mse_loss(y1[0], target[-1])=}'


		y0 = torch.cat([y0, y1], dim=0)

		# plt.scatter([y0[0,0].numpy(), y0[1,0].numpy()], [y0[0,1].numpy(), y0[1,1].numpy()])
		# plt.plot(target[:,0], target[:,1])
		# plt.show()
		# exit()

		return y0, self.output_length, target

	def __len__(self):

		# assert self.data.dim()==3

		if self.sample_axis=='trajs':
			return self.data.shape[0]*self.traj_repetition
		elif self.sample_axis=='timesteps':
			return (self.data.shape[1] - (self.input_length + self.output_length + self.input_length))
		else:
			raise ValueError('Sample axis not defined in data set not defined')

class Sequential_BiDirectional_TimeSeries_DataSet(Dataset):

	def __init__(self, data, input_length=1, output_length=2, sample_axis=None):

		assert data.dim()==3, f'Data.dim()={data.dim()} and not [trajs, steps, features]'
		assert type(input_length)==int
		assert type(output_length)==int
		assert input_length>=1
		assert output_length>=1

		if sample_axis is not None:
			assert sample_axis in ['trajs', 'timesteps'], f'Invalid axis ampling'

		self.input_length = input_length
		self.output_length = output_length
		self.output_length_samplerange = [output_length, output_length+1]

		T = data.shape[1]
		data = data[0]  # self.data_train.shape=(timeseries>1, t>1, F)
		
		# data_train = torch.stack(data.chunk(chunks=T // (2 * input_length + output_length-1), dim=0)[:-1])  # dropping the last, possibly wrong length time series sample
		data_train = torch.stack(data.chunk(chunks=T // (input_length + output_length-1), dim=0)[:-1])  # dropping the last, possibly wrong length time series sample
		data_train = torch.cat([data_train[:-1], data_train[1:,:input_length]], dim=1)


		# plt.plot(data_train[:3,:(-input_length),:3].flatten(0,1))
		# plt.plot(data_train[:3,:,:3].flatten(0,1))
		# plt.show()
		# exit()

		self.data = data_train

	def __getitem__(self, idx):

		'''
		Many short timeseries
		'''
		idx = idx % self.data.shape[0] # traj_repetition allows for "oversampling" of first dim -> restricting idx to original data shape
		traj = self.data[idx] # selecting trajectory

		'''
		y0 + input_length | output_length | y1 + input_length
		'''

		y0 = traj[:self.input_length]  # selecting corresponding starting point
		y1 = traj[-self.input_length:]  # selecting corresponding starting point

		target = traj  # snippet of trajectory
		assert F.mse_loss(y0, target[:self.input_length]) == 0, f'{F.mse_loss(y0, target[0])=}'
		assert F.mse_loss(y1,target[-self.input_length:])==0, f'{F.mse_loss(y1[0], target[-1])=}'

		y0 = torch.cat([y0, y1], dim=0)

		return y0, self.output_length, target

	def __len__(self):

		return self.data.shape[0]

class Sequential_TimeSeries_DataSet(Dataset):

	def __init__(self, data, input_length=1, output_length=2, sample_axis=None):

		assert data.dim()==3, f'Data.dim()={data.dim()} and not [trajs, steps, features]'
		assert type(input_length)==int
		assert type(output_length)==int
		assert input_length>=1
		assert output_length>=1

		if sample_axis is not None:
			assert sample_axis in ['trajs', 'timesteps'], f'Invalid axis ampling'

		self.input_length = input_length
		self.output_length = output_length
		self.output_length_samplerange = [output_length, output_length+1]

		T = data.shape[1]
		data = data[0]  # self.data_train.shape=(timeseries>1, t>1, F)
		data_train = torch.stack(data.chunk(chunks=T // (input_length + output_length-1), dim=0)[:-1])  # dropping the last, possibly wrong length time series sample
		self.data = data_train

	def __getitem__(self, idx):

		traj = self.data[idx] # selecting trajectory

		'''
		y0 + input_length | output_length | y1 + input_length
		'''

		y0 = traj[:self.input_length]  # selecting corresponding starting point


		target = traj  # snippet of trajectory
		assert F.mse_loss(y0, target[:self.input_length]) == 0, f'{F.mse_loss(y0, target[0])=}'
		# assert F.mse_loss(y1,target[-self.input_length:])==0, f'{F.mse_loss(y1[0], target[-1])=}'

		# y0 = torch.cat([y0, y1], dim=0)

		# if self.input_length == 1:
		# 	y0.squeeze_(0)

		return y0, self.output_length, target

	def __len__(self):

		return self.data.shape[0]

