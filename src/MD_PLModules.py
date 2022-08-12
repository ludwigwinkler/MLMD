from pathlib import Path
import sys, os, copy, math

filepath = Path(__file__).absolute()
cwd = filepath.parent
phd_path = Path(str(filepath).split('PhD')[0] + 'PhD')
mlmd_path = Path(str(filepath).split('MLMD')[0] + '/MLMD')
sys.path.append(str(mlmd_path))

from typing import Any, Dict, Union
import warnings
import numpy as np
from numbers import Number
from tqdm import tqdm

import matplotlib.pyplot as plt
import wandb

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.optim.lr_scheduler import ReduceLROnPlateau



from pytorch_lightning import LightningModule
from pytorch_lightning.utilities.cloud_io import load as pl_load
from pytorch_lightning.loggers import WandbLogger
from MLMD.src.MD_Models import MD_ODE, MD_Hamiltonian, MD_RNN, MD_LSTM, MD_ODE_SecOrder, MD_VAR
from MLMD.src.MD_Models import MD_BiDirectional_RNN, MD_BiDirectional_Hamiltonian, MD_BiDirectional_ODE, \
	MD_BiDirectional_LSTM
from MLMD.src.MD_Utils import Benchmark, Timer
from MLMD.src.MD_DataUtils import load_dm_data, QuantumMachine_DFT, Sequential_TimeSeries_DataSet, \
	Sequential_BiDirectional_TimeSeries_DataSet

from MLMD.experiments.ExportToXYZ import export_to_xyz
from MLMD.src.MD_AtomGeometry import Atom, compute_innermolecular_distances

from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked
patch_typeguard()

device = torch.device('cpu') if not torch.cuda.is_available() else torch.device('cuda:0')

class Interpolator(LightningModule):
	
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
		elif self.hparams.model == 'var':
			self.model = MD_VAR(hparams=self.hparams)
		else:
			exit(f'Wrong model: {self.hparams.model}')
	
	def load_weights(self):
		'''
		Custom checkpoint loading function that only loads the weights and, if chosen, the optim state
		PyTorch Lightnings load_from_checkpoint also changes the hyperparameters unfortunately, which we don't want
		'''
		
		ckpt_path = Path(__file__).parent / f'ckpt/{self.hparams.ckptname}.ckpt'
		
		ckpt_paths = [str.replace(str(ckpt_path), f'_T{self.hparams.output_length_train}', '_T5'),
		              str.replace(str(ckpt_path), f'_T{self.hparams.output_length_train}', '_T10'),
		              str.replace(str(ckpt_path), f'_T{self.hparams.output_length_train}', '_T20')]
		
		# for ckpt_path_ in ckpt_paths[:1]:
		for ckpt_path_ in [ckpt_path]:
			print()
			print(f'Looking for pretrained model at')
			print(f'{ckpt_path_}')
			if os.path.exists(ckpt_path_):
				print(" found", end='')
				
				old_param = copy.deepcopy(list(self.parameters())[5])
				
				self.load_state_dict(pl_load(ckpt_path_, map_location=device)['state_dict'], strict=True)
				assert (old_param - list(self.parameters())[5]).abs().sum() > 0.0
				print(' and loaded!')
			
			else:
				print(" but nothing found. :(")
			print()
	
	def on_fit_start(self):
		if self.hparams.load_weights:
			self.load_weights()
	
	@torch.no_grad()
	@typechecked
	def forward(self, t: TensorType[()], x: TensorType['BS', 'T', 'F']) -> TensorType['BS', -1, 'F']:
		'''
		x.shape = [BS, InputLength, F]
		out.shape = [BS, InputLength+OutputLength, F]
		'''
		
		return self.model(t, x)
	
	def training_step(self, batch, batch_idx):
		
		batch_y0, batch_t, batch_y = batch
		batch_pred = self.model(batch_t[0], batch_y0)
		
		batch_loss = self.model.criterion(batch_pred, batch_y)
		loss = batch_loss
		batch_loss = batch_loss.detach()
		
		with torch.no_grad():
			rescale = self.trainer.datamodule.unnormalize
			angle, dist = Atom(rescale(batch_y), self.hparams).compute_MD_geometry()
			pred_angle, pred_dist = Atom(rescale(batch_pred), self.hparams).compute_MD_geometry()
			
			if self.hparams.criterion == 'MSE':
				criterion = F.mse_loss
			elif self.hparams.criterion == 'MAE':
				criterion = F.l1_loss
			
			train_dist_loss = criterion(dist, pred_dist)
			train_angle_loss = torch.ones_like(train_dist_loss).fill_(-1.)  # criterion(angle, pred_angle)
		
		self.log_dict({'Train/' + self.hparams.criterion: batch_loss}, prog_bar=False)
		self.log_dict({'Train/t': batch_t[0]}, prog_bar=True)
		self.log_dict({'Train/AngleLoss': train_angle_loss, 'Train/DistLoss': train_dist_loss}, prog_bar=False)
		
		return {
		 'loss'                           : loss,
		 'Train/' + self.hparams.criterion: batch_loss,
		 'Train/AngleLoss'                : train_angle_loss,
		 'Train/DistLoss'                 : train_dist_loss}
	
	# def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int) -> None:
		
		# if self.hparams.output_length_sampling:
			# self.trainer.train_dataloader.dataset.sample_output_length()
	
	def training_epoch_end(self, outputs):
		train_loss = torch.stack([x['Train/' + self.hparams.criterion] for x in outputs]).mean()
		train_angle_loss = torch.stack([x['Train/AngleLoss'] for x in outputs]).mean()
		train_dist_loss = torch.stack([x['Train/DistLoss'] for x in outputs]).mean()
		self.log_dict({'Train/Loss': train_loss}, prog_bar=True)
		self.log_dict({'Train/EpochAngleLoss': train_angle_loss, 'Train/EpochDistLoss': train_dist_loss},
		              prog_bar=False)
	
	def validation_step(self, batch, batch_idx):
		batch_y0, batch_t, batch_y = batch
		
		batch_pred = self.model(batch_t[0], batch_y0)
		batch_loss = self.model.criterion(batch_pred, batch_y)
		
		with torch.no_grad():
			# rescale = self.trainer.datamodule.unnormalize
			rescale = lambda x: x
			pos, vel = torch.chunk(batch_y, chunks=2, dim=-1)
			pred_pos, pred_vel = torch.chunk(batch_pred, chunks=2, dim=-1)
			angle, dist = Atom(rescale(batch_y), self.hparams).compute_MD_geometry()
			pred_angle, pred_dist = Atom(rescale(batch_pred), self.hparams).compute_MD_geometry()
			
			if self.hparams.criterion == 'MSE':
				criterion = F.mse_loss
			elif self.hparams.criterion == 'MAE':
				criterion = lambda pred, target: (pred - target).abs().mean()
			
			val_dist_loss = criterion(dist, pred_dist)
			val_velocity_loss = criterion(vel, pred_vel)
			val_angle_loss = torch.ones_like(val_dist_loss).fill_(-1.)  # criterion(angle, pred_angle)
			
			for_loss = self.model.criterion(self.model.pred_for, batch_y, validate_args=False)
			back_loss = self.model.criterion(self.model.pred_back, batch_y, validate_args=False)
		
		return {
		 'Val/' + self.hparams.criterion    : batch_loss,
		 'Val/For' + self.hparams.criterion : for_loss,
		 'Val/Back' + self.hparams.criterion: back_loss,
		 'Val/AngleLoss'                    : val_angle_loss,
		 'Val/DistLoss'                     : val_dist_loss,
		 'Val/VelLoss'                      : val_velocity_loss,
		 'Val/t'                            : batch_t[0]}
	
	def validation_epoch_end(self, outputs):
		val_loss = torch.stack([x['Val/' + self.hparams.criterion] for x in outputs]).mean()
		val_for_loss = torch.stack([x['Val/For' + self.hparams.criterion] for x in outputs]).mean()
		val_back_loss = torch.stack([x['Val/Back' + self.hparams.criterion] for x in outputs]).mean()
		val_angle_loss = torch.stack([x['Val/AngleLoss'] for x in outputs]).mean()
		val_dist_loss = torch.stack([x['Val/DistLoss'] for x in outputs]).mean()
		val_vel_loss = torch.stack([x['Val/VelLoss'] for x in outputs]).mean()
		
		self.log_dict({'Val/Epoch' + self.hparams.criterion: val_loss, 'Val/t': self.hparams.output_length_val},
		              prog_bar=True)
		self.log_dict({
		               'Val/EpochFor' + self.hparams.criterion : val_for_loss,
		               'Val/EpochBack' + self.hparams.criterion: val_back_loss}, prog_bar=True)
		self.log_dict({
		               'Val/EpochAngleLoss': val_angle_loss,
		               'Val/EpochDistLoss' : val_dist_loss,
		               'Val/EpochVelLoss'  : val_vel_loss}, prog_bar=False)
	
	def configure_optimizers(self):
		if self.hparams.lr == 0:
			warnings.warn(f"Learning rate is {self.hparams.lr}")
		if self.hparams.optim == 'adam':
			optim = torch.optim.Adam(self.model.parameters(), lr=1e-3 if self.hparams.lr <= 0 else self.hparams.lr)
			schedulers = {
			 'scheduler'        : ReduceLROnPlateau(optim,
			                                        mode='min',
			                                        factor=0.5,
			                                        threshold=1e-3,
			                                        patience=1,
			                                        min_lr=1e-5,
			                                        verbose=True),
			 'monitor'          : 'Val/Epoch' + self.hparams.criterion,
			 'interval'         : 'epoch',
			 'reduce_on_plateau': True,
			 'frequency'        : 1,
			 'strict'           : True, }
		elif self.hparams.optim == 'sgd':
			optim = torch.optim.SGD(self.model.parameters(),
			                        lr=1e-2 if self.hparams.lr <= 0 else self.hparams.lr,
			                        momentum=0.8,
			                        dampening=0.6)
			schedulers = {
			 'scheduler'        : ReduceLROnPlateau(optim,
			                                        mode='min',
			                                        factor=0.1,
			                                        threshold=1e-3,
			                                        patience=2,
			                                        min_lr=1e-5,
			                                        verbose=True),
			 'monitor'          : 'Val/Epoch' + self.hparams.criterion,
			 'interval'         : 'epoch',
			 'reduce_on_plateau': True,
			 'frequency'        : 1,
			 'strict'           : True, }
		
		return [optim], [schedulers]
	
	def plot_interatomic_distances_histogram(self, dm):
		
		'''Predict trajectories'''
		y, pred = self.predict_batches(dm)
		
		'''Compute angles and velocities'''
		pred_dist = compute_innermolecular_distances(pred)
		true_dist = compute_innermolecular_distances(y)
		
		'''Flatten for dists'''
		pred_dist = pred_dist.flatten()[torch.randperm(pred_dist.numel())][:100000]
		true_dist = true_dist.flatten()[torch.randperm(true_dist.numel())][:100000]
		
		pred_dist = pred_dist[pred_dist != 0.0]
		true_dist = true_dist[true_dist != 0.0]
		
		dm.plot_interatomic_distances_histogram(true_dist,
		                                        pred_dist,
		                                        str=self.hparams.dataset_nicestr + f": T={self.hparams.output_length * 0.2}fs")
		
		if self.hparams.show:
			plt.show()
	
	def plot_speed_histogram(self, dm):
		
		y, pred = self.predict_batches(dm)
		
		pred_pos, pred_vel = pred.chunk(chunks=2, dim=-1)
		pos, vel = y.chunk(chunks=2, dim=-1)
		
		BS, T, F = vel.shape
		vel = vel.reshape(BS, T, -1, 3)
		pred_vel = pred_vel.reshape(BS, T, -1, 3)
		
		speed = vel.abs().sum(dim=-1)[:].flatten()
		pred_speed = pred_vel.abs().sum(dim=-1)[:].flatten()
		
		pred_speed = pred_speed.flatten()[torch.randperm(pred_speed.numel())][:100000]
		speed = speed.flatten()[torch.randperm(speed.numel())][:100000]
		
		dm.plot_speed_histogram(speed, pred_speed, str=self.hparams.dataset_nicestr)
		
		if self.hparams.show:
			plt.show()
	
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
			
			if i == 10:
				break
		
		'''Batched predictions [batch_size, timestpes, features]'''
		pred = torch.cat(pred_, dim=0).cpu().detach()
		y = torch.cat(y_, dim=0).cpu().detach()
		y0 = torch.cat(y0_, dim=0).cpu().detach()
		
		return y, pred
	
	@torch.no_grad()
	def predict_sequentially(self, dm=None, num_sequential_samples=None):
		
		self.eval()
		
		if dm is None:
			data_val = self.trainer.datamodule.val_dataloader().dataset.data
		elif dm is not None:
			# data_val = dm.val_dataloader().dataset.data
			# data_val = dm.train_dataloader().dataset.data
			data_val = dm.data_norm
		assert data_val.dim() == 3
		assert num_sequential_samples is None or type(num_sequential_samples) == int
		
		num_sequential_samples = np.min([100000 if num_sequential_samples is None else num_sequential_samples,
		                                 int(data_val.shape[1] * 0.9)])
		
		if 'bi' in self.hparams.model:
			'''
			Sequential DataSet chops up the continuous trajectory [T, F] into chunks of the appropriate prediction shape
			If we flatten the [BS, TS, F] minibatches we get back the continuous trajectory [T,F]
			'''
			data_set = Sequential_BiDirectional_TimeSeries_DataSet(data_val[0],
			                                                       input_length=self.hparams.input_length,
			                                                       output_length=self.hparams.output_length_val)
		else:
			data_set = Sequential_TimeSeries_DataSet(data_val[0],
			                                         input_length=self.hparams.input_length,
			                                         output_length=self.hparams.output_length_val, )
		
		dataloader = DataLoader(data_set,
		                        batch_size=100,
		                        sampler=torch.utils.data.SequentialSampler(data_set.data),
		                        drop_last=False)
		
		segments = num_sequential_samples // (self.hparams.input_length + self.hparams.output_length_val)
		input_length_ = np.tile(np.arange(0, self.hparams.input_length), segments)
		segment_offset = np.repeat(np.arange(0, segments) * (
			 self.hparams.input_length + self.hparams.output_length_val), self.hparams.input_length)
		t_y0 = input_length_ + segment_offset
		
		y_ = []
		y0_ = []
		pred_ = []
		total_length = 0
		pbar = tqdm(dataloader, total=num_sequential_samples)
		for y0, t, y in pbar:
			
			y0 = y0.to(device)
			t = t.to(device)
			self.model.to(device)
			
			batch_pred = self.forward(t[0], y0)  # [ BS, T, F]
			
			if 'bi' in self.hparams.model:
				pred_ += [*batch_pred[:, :-self.hparams.input_length]]
				y_ += [*y[:, :-self.hparams.input_length]]
			else:
				pred_ += [*batch_pred]
				y_ += [*y]
			y0_.append(y0[:, :(self.hparams.input_length)].flatten(0,
			                                                       1))  # add [BS*(input_length+output_length), Features]
			
			total_length += math.prod(batch_pred.shape[:2])
			pbar.set_description(desc=f"Samples: {num_sequential_samples} | Progress: {total_length / num_sequential_samples:.2f}")
			if total_length > num_sequential_samples:
				break
		
		pred = [pred__.cpu().detach() for pred__ in pred_]
		y = [y__.cpu().detach() for y__ in y_]
		
		# pred = torch.cat(pred_, dim=0).cpu().detach()
		# y = torch.cat(y_, dim=0).cpu().detach()
		y0 = torch.cat(y0_, dim=0).cpu().detach()
		
		assert F.mse_loss(y[0], data_val[0, :y[0].shape[0]]) == 0
		# if self.hparams.plot and self.hparams.show:
		# 	if dm is not None: dm.plot_sequential_prediction(y=y, y0=y0, t0=t_y0, pred=pred)
		# 	elif dm is None: self.trainer.datamodule.plot_sequential_prediction(y=y, y0=y0, t0=t_y0, pred=pred)
		
		self.train()
		return y, pred, y0, t_y0
	
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
		
		if self.hparams.plot and self.current_epoch >= 0:
			fig = self.trainer.datamodule.plot_sequential_prediction(*self.predict_sequentially())
			if isinstance(self.logger, WandbLogger):
				print('Saving image to WandB ...')
				self.logger.experiment.log({'Pred': wandb.Image(fig, caption="Val Prediction")})
			
			if self.hparams.show:
				print('Saving image to disk ...')
				fig.savefig(f"{cwd.parent}/visualization/plots/{self.hparams.ckptname}.png")
				plt.show()