import sys, os, inspect, copy, time
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Union
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

fontsize = 20
params = {	'font.size' : fontsize,
		'legend.fontsize': fontsize,
	  	'xtick.labelsize': fontsize,
	  	'ytick.labelsize': fontsize,
	  	'axes.labelsize': fontsize,
	  }

plt.rcParams.update(params)

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

# This is a test

torch.set_printoptions(precision=5, sci_mode=False)
np.set_printoptions(precision=5, suppress=True)

sys.path.append("/".join(os.getcwd().split("/")[:-1])) # experiments -> MLMD
sys.path.append("/".join(os.getcwd().split("/")[:-2])) # experiments -> MLMD -> PHD
# sys.path.append(os.getcwd())
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

# cwd = os.path.abspath(os.getcwd())
# os.chdir(cwd)

from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer, seed_everything
import wandb
from pytorch_lightning.loggers import WandbLogger

seed_everything(123)

from MLMD.src.MD_AtomGeometry import Atom, compute_innermolecular_distances
from MLMD.src.MD_PL_CallBacks import CustomModelCheckpoint
from MLMD.src.MD_Models import MD_ODE, MD_Hamiltonian, MD_RNN, MD_LSTM, MD_ODE_SecOrder
from MLMD.src.MD_Models import MD_BiDirectional_RNN, MD_BiDirectional_Hamiltonian, MD_BiDirectional_ODE, MD_BiDirectional_LSTM
from MLMD.src.MD_HyperparameterParser import HParamParser
from MLMD.src.MD_DataUtils import load_dm_data, QuantumMachine_DFT, Sequential_TimeSeries_DataSet, \
	Sequential_BiDirectional_TimeSeries_DataSet
from MLMD.src.MD_Utils import Benchmark, Timer


class Model(LightningModule):

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
			self.model = MD_ODE(hparams=self.hparams, scaling=scaling)
		elif self.hparams.model == 'ode2':
			self.model = MD_ODE_SecOrder(hparams=self.hparams, scaling=scaling)
		else:
			exit(f'Wrong model: {self.hparams.model}')

	def load_model_and_optim_state_dicts(self):
		'''
		Custom checkpoint loading function that only loads the weights and, if chosen, the optim state
		PyTorch Lightnings load_from_checkpoint also changes the hyperparameters unfortunately, which we don't want
		'''

		if self.hparams.load_pretrained:
			print(f"Looking for checkpoint ... ", end='')
			state_dict_path = f"{os.getcwd()}/ckpt/{hparams.ckptname}.state_dict"
			if os.path.exists(state_dict_path):
				print(f"Found at {state_dict_path} ... ", end='')
				state_dict = torch.load(state_dict_path, map_location=device)
				try:
					self.model.load_state_dict(state_dict, strict=True);
					print(f"and loaded!")
				except:
					print(f" but couldn't load.")

	def on_fit_start(self):

		if self.hparams.load_pretrained:
			self.load_model_and_optim_state_dicts()

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

		batch_loss = self.model.criterion(batch_pred, batch_y, mode='t')
		loss = batch_loss
		batch_loss = batch_loss.detach()
		
		self.log_dict({'Train/t': batch_t[0], 'Train/MSE': batch_loss}, prog_bar=True)

		return {'loss': loss, 'Train/MSE': batch_loss}

	def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:

		if self.hparams.output_length_sampling:
			self.trainer.train_dataloader.dataset.sample_output_length()

	def training_epoch_end(self, outputs):
		val_loss = torch.stack([x['Train/MSE'] for x in outputs]).mean()
		self.log('Train/EpochMSE', val_loss, prog_bar=True)

	def validation_step(self, batch, batch_idx):
		batch_y0, batch_t, batch_y = batch

		batch_pred = self.forward(batch_t[0], batch_y0)
		batch_loss = self.model.criterion(batch_pred, batch_y, mode='t')

		return {'Val/MSE': batch_loss, 'Val/t': batch_t[0]}

	def validation_epoch_end(self, outputs):
		val_loss = torch.stack([x['Val/MSE'] for x in outputs]).mean()
		self.log_dict({'Val/EpochMSE':val_loss, 'Val/t': self.hparams.output_length_val}, prog_bar=True)

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters())

	def plot_velocity_histogram(self):
		if self.hparams.dataset in ['benzene_dft.npz', 'ethanol_dft.npz', 'malonaldehyde_dft.npz']:
			batch = next(self.trainer.datamodule.val_dataloader().__iter__())
			batch_y0, batch_t, batch_y = batch
			pos, vel = batch_y.chunk(chunks=2, dim=-1)
			with torch.no_grad(): pred = self.forward(batch_t[0], batch_y0)
			pred_pos, pred_vel = pred.chunk(chunks=2, dim=-1)

			# print(f"{vel.shape=}")
			bs, t, dim = vel.shape

			vel = vel.reshape(bs, t, -1, 3)
			pred_vel = pred_vel.reshape(bs, t, -1, 3)
			num_atoms = pred_vel.shape[2]

			MB_dist = lambda a, x: (2 / np.pi) ** 0.5 * x ** 2 * np.exp(-x ** 2 / (2 * a ** 2)) / a ** 3

			pred_vel += torch.randn_like(pred_vel) * 0.001
			magn_vel = vel.abs().sum(dim=-1)[:].flatten()
			magn_pred_vel = pred_vel.abs().sum(dim=-1)[:, 0].flatten()

			hist_params = {'bins': 100, 'alpha': 0.25, 'density': True}
			x = np.linspace(0, 10, 100)
			plt.plot(x, MB_dist(1.5, x), lw=5, color='green', ls='--', label=r'$p_{MB}(x|a=1.5)$')
			plt.hist(magn_vel, **hist_params, color='red', label=r'$||\dot{q}(t)||$')
			plt.hist(magn_pred_vel, **hist_params, color='blue', label=r'$||\dot{\tilde{q}}(t)||$')
			plt.xlabel(r'Speed')
			plt.ylabel(r'Probability Density')
			plt.legend()
			plt.show()

	@torch.no_grad()
	def plot_interatomic_distances_histogram(self, dm):

		self.load_model_and_optim_state_dicts()

		dataloader = dm.val_dataloader()

		y_ = []
		y0_ = []
		pred_ = []
		for i, (y0, t, y) in enumerate(dataloader):
			y0 = y0.to(device)
			t = t.to(device)
			self.model.to(device)

			batch_pred = self.forward(t[0], y0)

			if 'bi' in self.hparams.model:
				pred_.append(batch_pred[:, :-self.hparams.input_length])
				y_.append(y[:, :-self.hparams.input_length])
			else:
				pred_.append(batch_pred)
				y_.append(y)

			y0_.append(y0[:, :(self.hparams.input_length)])

			if i==10: break

		'''Batched predictions [batch_size, timestpes, features]'''
		pred 	= torch.cat(pred_, dim=0).cpu().detach()
		y 	= torch.cat(y_, dim=0).cpu().detach()
		y0 	= torch.cat(y0_, dim=0).cpu().detach()

		assert F.mse_loss(pred[:,0,:], y[:,0,:])==0

		pred_dist = compute_innermolecular_distances(pred)
		true_dist = compute_innermolecular_distances(y)

		pred_dist = pred_dist.flatten()[torch.randperm(pred_dist.numel())][:100000]
		true_dist = true_dist.flatten()[torch.randperm(true_dist.numel())][:100000]

		pred_dist = pred_dist[pred_dist!=0.0]
		true_dist = true_dist[true_dist!=0.0]

		assert pred_dist.dim()==true_dist.dim()==1

		fig = plt.figure()

		plt.hist(pred_dist.numpy(), density=True, bins=100, color='red', alpha=0.5, label='Target')
		plt.hist(true_dist.numpy(), density=True, bins=100, color='blue', alpha=0.5, label='Predicted')
		plt.xlabel('$d[\AA]$')
		plt.ylabel('$p(d)$')
		plt.legend()
		plt.savefig(fname=f"{self.hparams.dataset[:-4]}_{self.hparams.model}.png", format='png')
		if self.hparams.show: plt.show()

	@torch.no_grad()
	def predict_batches(self, dm=None):
		'''
		dm: external datamodule
		'''

		if dm is None:
			dm = self.trainer.datamodule # the datamodule that was used for optimization

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
				pred_.append(batch_pred[:,self.hparams.input_length])
				y_.append(y[:, self.hparams.input_length])

			y0_.append(y0)

			if i == 10: break

		'''Batched predictions [batch_size, timestpes, features]'''
		pred = torch.cat(pred_, dim=0).cpu().detach()
		y = torch.cat(y_, dim=0).cpu().detach()
		y0 = torch.cat(y0_, dim=0).cpu().detach()

	@torch.no_grad()
	def predict_sequentially(self, dm=None):

		# print("Plotting ...")
		if dm is None:
			data_val = self.trainer.datamodule.val_dataloader().dataset.data
		elif dm is not None:
			# data_val = dm.val_dataloader().dataset.data
			# data_val = dm.train_dataloader().dataset.data
			data_val = dm.data_norm
		assert data_val.dim() == 3
		num_sequential_samples = np.min([500, data_val.shape[1]])

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
		for y0, t, y in tqdm(dataloader, desc='Predicting Sequentially'):

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
			y0_.append(y0[:, :(self.hparams.input_length)].flatten(0, 1)) # add [BS*(input_length+output_length), Features]
			total_length += pred_[-1].shape[0]
			if total_length>num_sequential_samples: break


		pred = torch.cat(pred_, dim=0).cpu().detach()
		y = torch.cat(y_, dim=0).cpu().detach()
		y0 = torch.cat(y0_, dim=0).cpu().detach()

		assert F.mse_loss(y, data_val[0, :y.shape[0]]) == 0
		if self.hparams.plot and self.hparams.show:
			if dm is not None: dm.plot_sequential_prediction(y, y0, t_y0, pred)
			elif dm is None: self.trainer.datamodule.plot_sequential_prediction(y, y0, t_y0, pred)

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
				input = torch.randn(1,2*hparams.input_length, data_val.shape[-1])
			else:
				input = torch.randn(1,hparams.input_length, data_val.shape[-1])

			timer = Timer()
			reps = 100
			for i in range(reps):
				_ = self.forward(self.hparams.output_length, input)
			time = timer.stop()
			print(f"{self.hparams.model}: {np.around(time)} s for {reps} single step predictions = {np.around(time/reps,5)}s/it")
		elif torch.cuda.is_available():

			device = torch.device('cuda')
			if 'bi_' in self.hparams.model:
				input = torch.randn(1, 2 * hparams.input_length, data_val.shape[-1], device=device)
			else:
				input = torch.randn(1, hparams.input_length, data_val.shape[-1], device=device)
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
			print(f"{self.hparams.model}: {np.around(np.sum(timings),5)} ms for {reps} single step predictions = {np.around(mean_time, 5)/1000}s/it")

	def on_fit_end(self):

		# checks that trainer has early_stopping_callback
		if len(self.trainer.callbacks) > 0:
			self.log('Val/MSE', self.trainer.callbacks[0].best_score, prog_bar=False)

		if isinstance(self.logger, WandbLogger): # saving state_dict to c
			state_dict_path = f"{os.getcwd()}/ckpt/{hparams.ckptname}.state_dict"
			if not os.path.exists(f"{os.getcwd()}/ckpt"): os.makedirs(f"{os.getcwd()}/ckpt")
			torch.save(self.model.state_dict(), state_dict_path)
			assert os.path.exists(state_dict_path) == True
			self.logger.experiment.save(state_dict_path)

		if self.hparams.plot and self.current_epoch>0:
			prediction_data = self.predict_sequentially()
			fig = self.trainer.datamodule.plot_sequential_prediction(*prediction_data)
			if isinstance(self.logger, WandbLogger):
				print('Saving image ...')
				self.logger.experiment.log({'Pred':wandb.Image(fig, caption="Val Prediction")})

			if self.hparams.show: plt.show()

hparams = HParamParser(logger=False, show=True, load_pretrained=False, fast_dev_run=False,
		       project='mlmd',
		       model='bi_lstm', num_layers=2, num_hidden_multiplier=5,
		       dataset=['hmc','benzene_dft.npz', 'toluene_dft.npz','hmc', 'keto_100K_0.2fs.npz', 'keto_300K_0.2fs.npz', 'keto_500K_0.2fs.npz'][2],
		       input_length=1, output_length=20, batch_size=200,
		       train_traj_repetition=20, max_epochs=2000)

dm = load_dm_data(hparams)

# print(f"{hparams.experiment=}")
# exit()

scaling = {'y_mu': dm.y_mu, 'y_std': dm.y_std, 'dy_mu': dm.dy_mu, 'dy_std': dm.y_std}
hparams.__dict__.update({'in_features': dm.y_mu.shape[-1]})
hparams.__dict__.update({'num_hidden': dm.y_mu.shape[-1]*hparams.num_hidden_multiplier})

model = Model(**vars(hparams))

# print(f"{model.summarize()}")
# model.measure_inference_speed(dm)
# exit()

if hparams.logger is True:
	os.system('wandb login --relogin afc4755e33dfa171a8419620e141ebeaeb8f27f5')
	logger = WandbLogger(project=hparams.project, entity='mlmd',name=hparams.experiment)
	hparams.__dict__.update({'logger': logger})

early_stop_callback = EarlyStopping(
	monitor='Val/EpochMSE', mode='min',
	patience=5,min_delta=0.000,
	verbose=True
)
# model_checkpoint_callback = CustomModelCheckpoint(	filepath=f"ckpt/{hparams.ckptname}",
# 					    		monitor='Val/MSE', mode='min',
# 							save_top_k=1)

trainer = Trainer.from_argparse_args(	hparams,
				     	# min_steps=1000,
				     	# max_steps=50,
				     	progress_bar_refresh_rate=10,
				     	callbacks=[early_stop_callback],
				     	# limit_train_batches=10,
				     	# limit_val_batches=10,
					val_check_interval=1.,
				     	# checkpoint_callback=model_checkpoint_callback,
				     	gpus=torch.cuda.device_count(),
					distributed_backend="ddp" if torch.cuda.device_count()>1 else None
				     	)

trainer.fit(model, datamodule=dm)
# model.predict_sequentially(dm)
# plt.show()
# model.plot_sequential_prediction(dm)
# plt.show()
