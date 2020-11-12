import sys, os, inspect
import numpy as np
import matplotlib.pyplot as plt
from typing import Any, Dict, Union
from tqdm import tqdm

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

fontsize = 20
params = {'legend.fontsize': fontsize,
	  # 'legend.handlelength': 2,
	  # 'text.usetex': True}

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
from MLMD.src.MD_Models import MD_ODE, MD_Hamiltonian, MD_RNN, MD_LSTM
from MLMD.src.MD_Models import MD_BiDirectional_RNN, MD_BiDirectional_Hamiltonian, MD_BiDirectional_ODE, MD_BiDirectional_LSTM
from MLMD.src.MD_HyperparameterParser import HParamParser
from MLMD.src.MD_DataUtils import load_dm_data, QuantumMachine_DFT, Sequential_TimeSeries_DataSet, \
	Sequential_BiDirectional_TimeSeries_DataSet


class Model(LightningModule):

	def __init__(self, scaling, **kwargs):
		super().__init__()
		self.save_hyperparameters()

		# self.scaling = scaling

		if self.hparams.model == 'bi_rnn':
			self.model = MD_BiDirectional_RNN(hparams=self.hparams, scaling=scaling)
		elif self.hparams.model == 'bi_hamiltonian':
			self.model = MD_BiDirectional_Hamiltonian(hparams=self.hparams, scaling=scaling)
		elif self.hparams.model == 'bi_ode':
			self.model = MD_BiDirectional_ODE(hparams=self.hparams, scaling=scaling)
		elif self.hparams.model == 'bi_lstm':
			self.model = MD_BiDirectional_LSTM(hparams=self.hparams, scaling=scaling)
		elif self.hparams.model == 'rnn':
			self.model = MD_RNN(hparams=self.hparams, scaling=scaling)
		elif self.hparams.model == 'lstm':
			self.model = MD_LSTM(hparams=self.hparams, scaling=scaling)
		elif self.hparams.model == 'hamiltonian':
			self.model = MD_Hamiltonian(hparams=self.hparams, scaling=scaling)
		elif self.hparams.model == 'ode':
			self.model = MD_ODE(hparams=self.hparams, scaling=scaling)
		else:
			exit(f'Wrong model: {self.hparams.model}')

		# any argument in the init() will be recorded, so we have to remove the tensor components or else the loggers will complain about gpu tensors
		del self.hparams['scaling']

	def load_model_and_optim_state_dicts(self):
		'''
		Custom checkpoint loading function that only loads the weights and, if chosen, the optim state
		PyTorch Lightnings load_from_checkpoint also changes the hyperparameters unfortunately, which we don't want
		'''

		print(f"Looking for checkpoint ... ", end='')
		ckptpath = f'ckpt/{self.hparams.ckptname}.ckpt'
		if os.path.exists(ckptpath):
			print(f"Found at {ckptpath} ... ", end='')

			load_weight = True
			load_optim = False

			ckpt = torch.load(ckptpath, map_location=device)
			if load_weight:
				self.load_state_dict(ckpt['state_dict'], strict=True)
				print(f"and loaded!")
			if load_optim: self.trainer.optimizers[0].load_state_dict(ckpt['optimizer_states'][0])

	def on_fit_start(self):

		if self.hparams.load_pretrained:
			self.load_model_and_optim_state_dicts()

	def forward(self, t, x):

		return self.model(t, x)

	def training_step(self, batch, batch_idx):

		batch_y0, batch_t, batch_y = batch
		batch_pred = self.forward(batch_t[0], batch_y0)

		batch_loss = self.model.criterion(batch_pred, batch_y, mode='t')
		
		# print(batch_t[0])
		self.log('Train/t', batch_t[0], prog_bar=True)
		self.log('Train/MSE', batch_loss, prog_bar=True)

		return {'loss': batch_loss, 'Train/MSE': batch_loss}

	def on_train_batch_end(self, outputs: Any, batch: Any, batch_idx: int, dataloader_idx: int) -> None:

		if self.hparams.output_length_sampling:
			self.trainer.train_dataloader.dataset.sample_output_length()

	def training_epoch_end(self, outputs):
		val_loss = torch.stack([x['Train/MSE'] for x in outputs]).mean()
		self.log('Train/Epoch_MSE', val_loss, prog_bar=True)

		if False and self.hparams.dataset in ['benzene_dft.npz', 'ethanol_dft.npz', 'malonaldehyde_dft.npz']:
			angle = torch.stack([x['Angle'] for x in outputs]).mean()
			dist = torch.stack([x['Dist'] for x in outputs]).mean()

			# self.plot_velocity_histogram()
			# self.plot_predictions()

	def validation_step(self, batch, batch_idx):
		batch_y0, batch_t, batch_y = batch

		batch_pred = self.forward(batch_t[0], batch_y0)
		batch_loss = self.model.criterion(batch_pred, batch_y, mode='t')

		if False and self.hparams.dataset in ['benzene_dft.npz', 'ethanol_dft.npz', 'malonaldehyde_dft.npz']:
			batch_pred_angles, batch_pred_distances = Atom(batch_pred, hparams).compute_MD_geometry()
			batch_angles, batch_distances = Atom(batch_y, hparams).compute_MD_geometry()

			batch_angle_loss = F.mse_loss(batch_pred_angles, batch_angles)
			batch_distance_loss = F.mse_loss(batch_pred_distances, batch_distances)

		return {'Val/MSE': batch_loss, 'Val/t': batch_t[0]}

	def validation_epoch_end(self, outputs):
		val_loss = torch.stack([x['Val/MSE'] for x in outputs]).mean()
		self.log_dict({'Val/MSE':val_loss, 'Val/t': self.hparams.output_length_val}, prog_bar=True)

		if False and self.hparams.dataset in ['benzene_dft.npz', 'ethanol_dft.npz', 'malonaldehyde_dft.npz']:
			angle = torch.stack([x['Angle'] for x in outputs]).mean()
			dist = torch.stack([x['Dist'] for x in outputs]).mean()

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters())

	def plot_predictions(self):
		batch = next(self.trainer.datamodule.val_dataloader().__iter__())
		batch_y0, batch_t, batch_y = batch
		pos, vel = batch_y.chunk(chunks=2, dim=-1)
		with torch.no_grad(): pred = self.forward(batch_t[0], batch_y0)
		pred_pos, pred_vel = pred.chunk(chunks=2, dim=-1)
		bs, t, dim = vel.shape

		vel = vel.reshape(bs, t, -1, 3)
		pos = pos.reshape(bs, t, -1, 3)
		pred_vel = pred_vel.reshape(bs, t, -1, 3)
		pred_pos = pred_pos.reshape(bs, t, -1, 3)
		num_atoms = pred_vel.shape[2]

		# print(f"{pred_vel.shape=}")

		fig, axs = plt.subplots(2, 2, sharex=True)
		axs = axs.flatten()

		fig.suptitle('Vel')
		axs[0].plot(pred_vel[0, :, 0, :], ls='--')
		axs[1].plot(pred_vel[0, :, 1, :], ls='--')
		axs[2].plot(pred_vel[0, :, 2, :], ls='--')

		axs[0].plot(vel[0, :, 0, :])
		axs[1].plot(vel[0, :, 1, :])
		axs[2].plot(vel[0, :, 2, :])
		plt.show()

		fig, axs = plt.subplots(2, 2, sharex=True)
		axs = axs.flatten()

		fig.suptitle('Pos')
		axs[0].plot(pred_pos[0, :, 0, :], ls='--')
		axs[1].plot(pred_pos[0, :, 1, :], ls='--')
		axs[2].plot(pred_pos[0, :, 2, :], ls='--')

		axs[0].plot(pos[0, :, 0, :])
		axs[1].plot(pos[0, :, 1, :])
		axs[2].plot(pos[0, :, 2, :])

		plt.show()

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
	def predict_sequentially(self, dm=None):

		# print("Plotting ...")
		if dm is None:
			data_val = self.trainer.datamodule.val_dataloader().dataset.data
		elif dm is not None:
			# data_val = dm.val_dataloader().dataset.data
			# data_val = dm.train_dataloader().dataset.data
			data_val = dm.data_norm
		assert data_val.dim() == 3
		num_sequential_samples = np.min([100000000, data_val.shape[1]])

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
		dm.save_prediction(pred, y)


		return pred, y, y0, t_y0, num_sequential_samples

	@torch.no_grad()
	def plot_sequential_prediction(self, dm=None):
		'''
		pred: the prediction by the neural network
		'''
		pred, y, y0, t_y0, num_sequential_samples = self.predict_sequentially(dm)

		colors=['r', 'g', 'b']

		fig = plt.figure(figsize=(20,10))

		plt.plot(y[:num_sequential_samples, -3], color=colors[0], label='Data')
		plt.plot(y[:num_sequential_samples, -2], color=colors[1])
		plt.plot(y[:num_sequential_samples, -1], color=colors[2])
		plt.plot(pred[:num_sequential_samples, -3], ls='--', color=colors[0], label='Prediction')
		plt.plot(pred[:num_sequential_samples, -2], ls='--', color=colors[1])
		plt.plot(pred[:num_sequential_samples, -1], ls='--', color=colors[2])

		t_y0 = t_y0[:y0.shape[0]]

		plt.scatter(t_y0, y0[:t_y0.shape[0], -3], color=colors[0], label='Initial and Final Conditions')
		plt.scatter(t_y0, y0[:t_y0.shape[0], -2], color=colors[1], label='Initial and Final Conditions')
		plt.scatter(t_y0, y0[:t_y0.shape[0], -1], color=colors[2], label='Initial and Final Conditions')
		plt.xlabel('t')
		plt.ylabel('$q(t)$')
		plt.title(self.hparams.dataset)
		plt.grid()
		# plt.xticks(np.arange(0,t_y0.max()))
		plt.xlim(0, t_y0.max())
		plt.legend()

		return fig

	def on_fit_end(self):

		self.log('Val/MSE', self.trainer.callbacks[0].best_score, prog_bar=False)

		if self.hparams.plot:
			fig = self.plot_sequential_prediction()
			if isinstance(self.logger, WandbLogger):
				print('Saving image ...')
				self.logger.experiment.log({'Pred':wandb.Image(fig, caption="Val Prediction")})

			if self.hparams.show: plt.show()

hparams = HParamParser(logname='MD', logger=False, show=True, load_pretrained=True,
		       model='bi_lstm', dataset=['toluene_dft.npz','hmc', 'keto_300K_0.2fs.npz', 'keto_500K_0.2fs.npz'][0], batchsize=100,
		       input_length=1, output_length_train=11, output_length_val=31)

dm = load_dm_data(hparams)
exit()
scaling = {'y_mu': dm.y_mu, 'y_std': dm.y_std, 'dy_mu': dm.dy_mu, 'dy_std': dm.y_std}
hparams.__dict__.update({'in_features': dm.y_mu.shape[-1]})
hparams.__dict__.update({'num_hidden': dm.y_mu.shape[-1]*10})

model = Model(scaling, **vars(hparams))

model.load_model_and_optim_state_dicts()
model.predict_sequentially(dm)
# model.plot_sequential_prediction(dm)
# plt.show()

exit()

if hparams.logger is True:
	os.system('wandb login --relogin afc4755e33dfa171a8419620e141ebeaeb8f27f5')
	logger = WandbLogger(project='SeparateIntegrationTimes', entity='mlmd',name=hparams.logname)
	hparams.__dict__.update({'logger': logger})

early_stop_callback = EarlyStopping(
	monitor='Val/MSE', mode='min',
	patience=3,min_delta=0.005,
	verbose=True
)
model_checkpoint_callback = CustomModelCheckpoint(	filepath=f"ckpt/{hparams.ckptname}",
					    		monitor='Val/MSE', mode='min',
							save_top_k=1)

# model = model.load_from_checkpoint(checkpoint_path=f'ckpt/{hparams.ckptname}.ckpt', scaling=scaling)
# ckpt = torch.load(f'ckpt/{hparams.ckptname}.ckpt')

trainer = Trainer.from_argparse_args(	hparams,
				     	max_epochs=2000,
				     	# min_steps=1000,
				     	# max_steps=50,
				     	progress_bar_refresh_rate=5,
				     	callbacks=[early_stop_callback],
				     	# limit_train_batches=10,
				     	# limit_val_batches=10,
					val_check_interval=1.,
				     	fast_dev_run=False,
				     	checkpoint_callback=model_checkpoint_callback,
				     	gpus=torch.cuda.device_count(),
					distributed_backend="ddp" if torch.cuda.device_count()>1 else None
				     	)

trainer.fit(model, datamodule=dm)

