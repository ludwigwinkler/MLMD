import sys, os, inspect, copy, time
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Union
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau

fontsize = 30
params = {'font.size': fontsize,
	  'legend.fontsize': fontsize,
	  'xtick.labelsize': fontsize,
	  'ytick.labelsize': fontsize,
	  'axes.labelsize': fontsize,
	  'figure.figsize': (10, 10),
	  'text.usetex': True,
	  'mathtext.fontset': 'stix',
	  'font.family': 'STIXGeneral'
	  }

plt.rcParams.update(params)

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

# This is a test

torch.set_printoptions(precision=5, sci_mode=False)
np.set_printoptions(precision=5, suppress=True)

sys.path.append("/".join(os.getcwd().split("/")[:-1]))  # experiments -> MLMD
sys.path.append("/".join(os.getcwd().split("/")[:-2]))  # experiments -> MLMD -> PHD
# sys.path.append(os.getcwd())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# cwd = os.path.abspath(os.getcwd())
# os.chdir(cwd)

from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
import wandb
from pytorch_lightning.loggers import WandbLogger

from MLMD.src.MD_AtomGeometry import Atom, compute_innermolecular_distances
from MLMD.src.MD_PL_CallBacks import CustomModelCheckpoint
from MLMD.src.MD_Models import MD_ODE, MD_Hamiltonian, MD_RNN, MD_LSTM, MD_ODE_SecOrder
from MLMD.src.MD_Models import MD_BiDirectional_RNN, MD_BiDirectional_Hamiltonian, MD_BiDirectional_ODE, MD_BiDirectional_LSTM
from MLMD.src.MD_HyperparameterParser import Interpolation_HParamParser
from MLMD.src.MD_DataUtils import load_dm_data, QuantumMachine_DFT, Sequential_TimeSeries_DataSet, \
	Sequential_BiDirectional_TimeSeries_DataSet
from MLMD.src.MD_Utils import Benchmark, Timer

class MLMD_Model(LightningModule):

	def __init__(self, **kwargs):
		super().__init__()

		self.save_hyperparameters()

		if self.hparams.model == 'bi_rnn':
			self.model = MD_BiDirectional_RNN(hparams=self.hparams)
		elif self.hparams.model == 'bi_hnn':
			self.model = MD_BiDirectional_Hamiltonian(hparams=self.hparams)
		elif self.hparams.model == 'bi_ode':
			self.model = MD_BiDirectional_ODE(hparams=self.hparams)
		elif self.hparams.model == 'bi_lstm':
			self.model = MD_BiDirectional_LSTM(hparams=self.hparams)
		elif self.hparams.model == 'rnn':
			self.model = MD_RNN(hparams=self.hparams)
		elif self.hparams.model == 'lstm':
			self.model = MD_LSTM(hparams=self.hparams)
		elif self.hparams.model == 'hnn':
			# self.model = MD_Hamiltonian(hparams=self.hparams)
			self.model = MD_Hamiltonian(hparams=self.hparams)
		elif self.hparams.model == 'ode':
			self.model = MD_ODE(hparams=self.hparams)
		elif self.hparams.model == 'ode2':
			self.model = MD_ODE_SecOrder(hparams=self.hparams)
		else:
			exit(f'Wrong model: {self.hparams.model}')

	def load_weights(self):
		'''
		Custom checkpoint loading function that only loads the weights and, if chosen, the optim state
		PyTorch Lightnings load_from_checkpoint also changes the hyperparameters unfortunately, which we don't want
		'''
		if self.hparams.load_pretrained:
			ckpt_path = f"{os.getcwd()}/ckpt/{self.hparams.ckptname}.ckpt"
			print(f'Looking for pretrained model at {ckpt_path} ...', end='')
			if os.path.exists(ckpt_path):
				print(" found", end='')
				self.load_from_checkpoint(ckpt_path)
				print(" and loaded!")
			else:
				print(" but nothing found. :(")

	def on_fit_start(self):
		self.load_weights()

	def forward(self, t, x):
		'''
		x.shape = [BS, InputLength, F]
		out.shape = [BS, InputLength+OutputLength, F]
		'''
		out = self.model(t, x)
		return out

	def training_step(self, batch, batch_idx):

		batch_y0, batch_t, batch_y = batch
		batch_pred = self.forward(batch_t[0], batch_y0)

		batch_loss = self.model.criterion(batch_pred, batch_y)
		loss = batch_loss
		batch_loss = batch_loss.detach()

		# if self.trainer.current_epoch==5:
		# 	plt.plot(batch_y[0].numpy(), ls='--', c='r', label='y')
		# 	plt.plot(self.model.out[0].detach().numpy(), ls='-', c='g', label='pred')
		# 	# plt.plot(self.model.pred_back[0].detach().numpy(), ls='-', c='b', label='pred back')
		# 	plt.legend()
		# 	plt.show()
		#
		# 	plt.plot(self.forward(1000, Tensor([1,0]).reshape(1,1,-1)).squeeze(0).detach().numpy(), label='pred')
		# 	plt.show()
		# 	exit()

		self.log_dict({'Train/t': batch_t[0]}, prog_bar=True)
		self.log_dict({'Train/MAE': batch_loss}, prog_bar=False)

		return {'loss': loss, 'Train/MAE': batch_loss}

	def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:

		if self.hparams.output_length_sampling:
			self.trainer.train_dataloader.dataset.sample_output_length()

	def training_epoch_end(self, outputs):
		val_loss = torch.stack([x['Train/MAE'] for x in outputs]).mean()
		self.val_loss = val_loss
		self.log('Train/EpochMAE', val_loss, prog_bar=True)

	def validation_step(self, batch, batch_idx):
		batch_y0, batch_t, batch_y = batch

		batch_pred = self.forward(batch_t[0], batch_y0)
		batch_loss = self.model.criterion(batch_pred, batch_y)

		return {'Val/MAE': batch_loss, 'Val/t': batch_t[0]}

	def validation_epoch_end(self, outputs):
		self.val_loss = torch.stack([x['Val/MAE'] for x in outputs]).mean()
		output_length_val = outputs[0]['Val/t']
		self.log_dict({'Val/EpochMAE': self.val_loss, 'Val/t': output_length_val}, prog_bar=True)

	def configure_optimizers(self):
		optim = torch.optim.Adam(self.model.parameters(), lr=5e-4)
		schedulers = {
			'scheduler': ReduceLROnPlateau(optim, mode='min', factor=0.5, threshold=1e-4, patience=2, min_lr=1e-5, verbose=False),
			'monitor': 'Val/EpochMAE',
			'interval': 'epoch',
			'reduce_on_plateau': True,
			'frequency': 1,
			'strict': True,
		}

		return [optim], [schedulers]

	def plot_interatomic_distances_histogram(self, dm):

		y, pred = self.predict_batches(dm)

		pred_dist = compute_innermolecular_distances(pred)
		true_dist = compute_innermolecular_distances(y)

		pred_dist = pred_dist.flatten()[torch.randperm(pred_dist.numel())][:100000]
		true_dist = true_dist.flatten()[torch.randperm(true_dist.numel())][:100000]

		pred_dist = pred_dist[pred_dist != 0.0]
		true_dist = true_dist[true_dist != 0.0]

		dm.plot_interatomic_distances_histogram(true_dist, pred_dist, str=self.hparams.dataset_nicestr)

		if self.hparams.show: plt.show()

	def plot_speed_histogram(self, dm):

		y, pred = self.predict_batches(dm)

		pred_pos, pred_vel = pred.chunk(chunks=2, dim=-1)
		pos, vel = pred.chunk(chunks=2, dim=-1)

		BS, T, F = vel.shape
		vel = vel.reshape(BS, T, -1, 3)
		pred_vel = pred_vel.reshape(BS, T, -1, 3)

		speed = vel.abs().sum(dim=-1)[:].flatten()
		pred_speed = pred_vel.abs().sum(dim=-1)[:].flatten()

		pred_speed = pred_speed.flatten()[torch.randperm(pred_speed.numel())][:100000]
		speed = speed.flatten()[torch.randperm(speed.numel())][:100000]

		dm.plot_speed_histogram(speed, pred_speed, str=self.hparams.dataset_nicestr)

		if self.hparams.show: plt.show()

	@torch.no_grad()
	def predict_batches(self, dm=None):
		'''
		dm: external datamodule
		output: [BS, TS, Features]
		'''

		if dm is None:
			dm = self.trainer.datamodule  # the datamodule that was used for optimization

		y_ = []
		y0_ = []
		pred_ = []
		for i, (y0, t, y) in enumerate(dm.val_dataloader()):
			y0 = y0.to(device)
			t = t.to(device)
			self.model.to(device)

			batch_pred = self.forward(t[0], y0)

			if 'bi' in self.hparams.model:
				pred_.append(batch_pred[:, self.hparams.input_length:-self.hparams.input_length])
				y_.append(y[:, self.hparams.input_length:-self.hparams.input_length])
			else:
				pred_.append(batch_pred[:, :self.hparams.input_length])
				y_.append(y[:, :self.hparams.input_length])

			y0_.append(y0)

			if i == 50: break

		'''Batched predictions [batch_size, timestpes, features]'''
		pred = torch.cat(pred_, dim=0).cpu().detach()
		y = torch.cat(y_, dim=0).cpu().detach()
		y0 = torch.cat(y0_, dim=0).cpu().detach()

		return y, pred

	@torch.no_grad()
	def predict_sequentially(self, dm=None):


		if dm is None:
			assert type(self.trainer.datamodule.val_dataloader(), Sequential_TimeSeries_DataSet) or type(self.trainer.datamodule.val_dataloader(), Sequential_BiDirectional_TimeSeries_DataSet)
			data_val = self.trainer.datamodule.val_dataloader().dataset.data
		elif dm is not None:
			# data_val = dm.val_dataloader().dataset.data
			# data_val = dm.train_dataloader().dataset.data
			data_val = dm.data_norm
		assert data_val.dim() == 3
		num_sequential_samples = np.min([500, data_val.shape[1]])

		# print(f"predict_sequentially: {data_val.shape=}")

		if 'bi' in self.hparams.model:
			'''
			Sequential DataSet chops up the continuous trajectory [T, F] into chunks of the appropriate prediction shape
			If we flatten the [BS, TS, F] minibatches we get back the continuous trajectory [T,F]
			'''
			data_set = Sequential_BiDirectional_TimeSeries_DataSet(data_val[0], input_length=self.hparams.input_length,
									       output_length=self.hparams.output_length_val)
		else:
			data_set = Sequential_TimeSeries_DataSet(data_val[0], input_length=self.hparams.input_length,
								 output_length=self.hparams.output_length_val)

		dataloader = DataLoader(data_set, batch_size=100, sampler=torch.utils.data.SequentialSampler(data_set.data))

		segments = num_sequential_samples // (self.hparams.input_length + self.hparams.output_length_val)
		input_length_ = np.tile(np.arange(0, self.hparams.input_length), segments)
		segment_offset = np.repeat(np.arange(0, segments) * (self.hparams.input_length + self.hparams.output_length_val),
					   self.hparams.input_length)
		t_y0 = input_length_ + segment_offset

		y_ = []
		y0_ = []
		pred_ = []
		total_length = 0
		# for y0, t, y in tqdm(dataloader, desc='Predicting Sequentially'):
		for y0, t, y in dataloader:

			y0 = y0.to(device)
			t = t.to(device)
			self.model.to(device)

			batch_pred = self.forward(t[0], y0)

			if 'bi' in self.hparams.model:
				pred_.append(batch_pred[:, :-self.hparams.input_length].flatten(0, 1))
				y_.append(y[:, :-self.hparams.input_length].flatten(0, 1))
			else:
				pred_.append(batch_pred.flatten(0, 1))
				y_.append(y.flatten(0, 1))
			y0_.append(y0[:, :(self.hparams.input_length)].flatten(0, 1))  # add [BS*(input_length+output_length), Features]
			total_length += pred_[-1].shape[0]
			if total_length > num_sequential_samples: break

		pred = torch.cat(pred_, dim=0).cpu().detach()
		y = torch.cat(y_, dim=0).cpu().detach()
		y0 = torch.cat(y0_, dim=0).cpu().detach()

		assert F.mse_loss(y, data_val[0, :y.shape[0]]) == 0
		# if self.hparams.plot and self.hparams.show:
		# 	if dm is not None:
		# 		dm.plot_sequential_prediction(y, y0, t_y0, pred)
		# 	elif dm is None:
		# 		self.trainer.datamodule.plot_sequential_prediction(y, y0, t_y0, pred)

		return y, y0, t_y0, pred

	@torch.no_grad()
	def measure_inference_speed(self, dm):

		if dm is None:
			data_val = self.trainer.datamodule.val_dataloader().dataset.data
		elif dm is not None:
			data_val = dm.val_dataloader().dataset.data

		if not torch.cuda.is_available():
			print('CPU Inference')

			if 'bi_' in self.hparams.model:
				input = torch.randn(1, 2 * self.hparams.input_length, data_val.shape[-1])
			else:
				input = torch.randn(1, self.hparams.input_length, data_val.shape[-1])

			timer = Timer()
			reps = 100
			for i in range(reps):
				_ = self.forward(self.hparams.output_length, input)
			time = timer.stop()
			print(f"{self.hparams.model}: {np.around(time)} s for {reps} single step predictions = {np.around(time / reps, 5)}s/it")
		elif torch.cuda.is_available():

			device = torch.device('cuda')
			if 'bi_' in self.hparams.model:
				input = torch.randn(1, 2 * self.hparams.input_length, data_val.shape[-1], device=device)
			else:
				input = torch.randn(1, self.hparams.input_length, data_val.shape[-1], device=device)
			self.model.to(device)
			starter, ender = torch.cuda.Event(enable_timing=True), torch.cuda.Event(enable_timing=True)

			timer = Timer()
			reps = 100

			timings = np.zeros((reps, 1))
			# GPU-WARM-UP
			for _ in range(10):
				_ = self.forward(self.hparams.output_length, input)
			# MEASURE PERFORMANCE
			print(f"GPU Inference")
			with torch.no_grad():
				for rep in range(reps):
					starter.record()
					_ = self.forward(self.hparams.output_length, input)
					ender.record()
					# WAIT FOR GPU SYNC
					torch.cuda.synchronize()
					curr_time = starter.elapsed_time(ender)
					timings[rep] = curr_time
			mean_time = np.sum(timings) / reps
			std_time = np.std(timings)
			print(f"{self.hparams.model}: {np.around(np.sum(timings), 5)} ms for {reps} single step predictions = {np.around(mean_time, 5) / 1000}s/it")

	def on_fit_end(self):

		# state_dict_path = f"{os.getcwd()}/ckpt/{self.hparams.ckptname}.ckpt"
		# if not os.path.exists(f"{os.getcwd()}/ckpt"): os.makedirs(f"{os.getcwd()}/ckpt")
		# torch.save(self.model.state_dict(), state_dict_path)


		# if not self.hparams.save_weights: os.remove(state_dict_path)
		ckpt_path = f"{os.getcwd()}/ckpt/{self.hparams.ckptname}.ckpt"
		self.trainer.save_checkpoint(filepath=ckpt_path, weights_only=True)

		if isinstance(self.logger, WandbLogger):  # saving state_dict to cloud
			assert os.path.exists(ckpt_path)
			self.logger.experiment.save(ckpt_path)

		if self.hparams.plot and self.current_epoch > 0 and (type(self.trainer.datamodule.val_dataloader()) == Sequential_TimeSeries_DataSet or type(self.trainer.datamodule.val_dataloader()) == Sequential_BiDirectional_TimeSeries_DataSet):
			prediction_data = self.predict_sequentially()
			fig = self.trainer.datamodule.plot_sequential_prediction(*prediction_data)
			if isinstance(self.logger, WandbLogger):
				print('Saving image ...')
				self.logger.experiment.log({'Pred': wandb.Image(fig, caption="Val Prediction")})

			if self.hparams.show: plt.show()