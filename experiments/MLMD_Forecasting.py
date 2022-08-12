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
Scalar = torch.scalar_tensor

# This is a test

torch.set_printoptions(precision=5, sci_mode=False)
np.set_printoptions(precision=5, suppress=True)

sys.path.append("/".join(os.getcwd().split("/")[:-1]))  # experiments -> MLMD
sys.path.append("/".join(os.getcwd().split("/")[:-2]))  # experiments -> MLMD -> PHD

from MLMD.src.MD_SimMD import CosineDynamicalSystem, DoubleCosineDynamicalSystem
from MLMD.src.MLMD_ForecastingModel import MLMD_Model
from MLMD.src.MD_HyperparameterParser import Forecasting_HParamParser
from MLMD.src.MD_DataUtils import MLMD_Trajectory_DataSet, MLMD_Trajectory, MLMD_VariableTrajectorySegment_DataSet

from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything
import wandb
from pytorch_lightning.loggers import WandbLogger


class MLMD:

	def __init__(self, hparams):

		self.hparams = hparams

		self.mlmd = MLMD_Model(**vars(hparams))
		if self.hparams.dataset=='cosine':
			self.simmd = CosineDynamicalSystem(dt=self.hparams.dt, freq=self.hparams.cosine_freq)
		elif self.hparams.dataset == 'doublecosine':
			self.simmd = DoubleCosineDynamicalSystem(dt=self.hparams.dt, freq=self.hparams.cosine_freq)

		self.traj = MLMD_Trajectory(hparams=hparams)

		self.logger = hparams.logger if hparams.logger is not None else None

		self.control_every_n_steps = max(self.hparams.output_length_train, self.hparams.output_length_val)
		self.diff_threshold = 0.05

		self.forecasting_mode = ['mlmd', 'simmd']

	@property
	def t(self):
		return len(self.traj) * self.hparams.dt

	def mode(self, mode):
		assert mode in ['mlmd', 'simmd']
		self.forecasting_mode = mode

	def collect_init_traj(self, init_T=None):
		'''
		Collects the initial data set by running simmd and stores it in self.traj
		'''
		init_T_ = self.hparams.init_T if init_T is None else init_T
		self.simmd_forecast(init_T_, cat=True)
		assert self.traj.traj.shape==(init_T_, self.hparams.in_features), f"{self.traj.traj.shape=}"

	def setup_train_dataset(self):
		'''
		Checks and updates data set such that normalization and the conversion between normalized and unnormalized is proper
		Takes traj and creates a data set from it on which self.mlmd is trained on
		'''

		self.dm = MLMD_VariableTrajectorySegment_DataSet(self.hparams, segments=self.traj.get_simmd_traj_segments())
		self.dm.prepare_data()
		self.dm.setup()

		return self.dm

	def train_mlmd(self):
		'''
		Train the mlmd networks on the trajectory
		'''

		print()
		print()
		print(f"Retraining at step {len(self.traj)}")

		self.mlmd.zero_grad()  # required to initiate a clean new setup the next time

		early_stop_callback = EarlyStopping(monitor='Val/Epoch' + self.hparams.criterion, mode='min',
						    patience=5, min_delta=0.0001,
						    verbose=True)

		trainer = Trainer.from_argparse_args(self.hparams,
						     # min_steps=1000,
						     # max_steps=50,
						     progress_bar_refresh_rate=10,
						     # callbacks=[early_stop_callback],
						     # limit_train_batches=10,
						     # limit_val_batches=10,
						     num_sanity_val_steps=0,
						     val_check_interval=1.,
						     gpus=torch.cuda.device_count(),
						     distributed_backend="ddp" if torch.cuda.device_count() > 1 else None,
						     weights_summary=None
						     )

		print("#"*100)

		dm = self.setup_train_dataset()
		self.mlmd.model.data_mean = self.dm.data_mean
		self.mlmd.model.data_std = self.dm.data_std

		trainer.fit(self.mlmd, datamodule=dm)

		if self.hparams.plot:
			dict_figs = self.plot_traj_prediction()
			if isinstance(self.logger, WandbLogger):
				print('Saving image ...')
				for caption, fig in dict_figs:
					self.logger.experiment.log({caption: wandb.Image(fig, caption=caption)})
				# self.logger.experiment.log({'Val Forecast Spectrum': wandb.Image(fig_spectrum_pred, caption="Val Forecast Spectrum")})

			if self.hparams.show: plt.show()

		print("#" * 100)

	def retrain_mlmd_required(self) -> bool:
		'''
		Compares MLMD forecasting against SimMD forecasting
		'''

		target = self.simmd(T=self.hparams.output_length_val, t0=self.t)
		assert target.shape[0]==(self.hparams.output_length_val)
		pred = self.mlmd.model.forecast(T=self.hparams.output_length_val, x0=target[:1])
		assert target.shape==pred.shape, f"{target.shape=} != {pred.shape=}"

		if torch.isnan(target).any():
			print(f"{target=}")
			exit()
		if torch.isnan(pred).any():
			print(f"{pred=}")
			exit()

		diff = self.mlmd.model.criterion(pred, target)

		retrain = True if diff >= self.diff_threshold else False

		if retrain:
			print(f"Checking Performance at step {len(self.traj)}: {diff:.3f}>={self.diff_threshold}")
		else:
			print(f"Checking Performance at step {len(self.traj)}: {diff:.3f}<={self.diff_threshold}")

		# fig = self.dm.plot_sequential_prediction(y=target, pred=pred)
		# plt.show()

		if retrain and self.forecasting_mode == 'mlmd':
			self.traj.remove_last_mlmd_forecast()

		return retrain

	def simmd_forecast(self, T, cat=True):

		if self.traj.traj is None:	traj = self.simmd(T=T, x0=None, t0=0) # init trajectory
		else:				traj = self.simmd(T=T, x0=self.traj.traj[-1:], t0=self.t)
		if cat: self.traj += (traj, 'simmd')
		return traj

	def mlmd_forecast(self, T, cat=False):
		x = self.dm.normalize(self.traj.traj[-1:])
		traj = self.mlmd.model.forecast(T=T, x0=x)
		traj = self.dm.unnormalize(traj)
		if cat:
			self.mode('mlmd')
			self.traj += (traj, 'mlmd')
		return traj

	def plot_traj_prediction(self):

		steps = int(8*2*np.pi/self.hparams.dt)
		target = self.simmd(T=steps, x0=self.traj.traj[-1:], t0=self.t)
		pred_x_init = target[:1] if 'lstm' in self.hparams.model else target[:1]
		pred = self.mlmd.model.forecast(T=steps, x0=pred_x_init, t0=self.t)

		assert (target.dim()==pred.dim()==2)

		target.squeeze_(0)
		pred.squeeze_(0)

		if not hasattr(self, 'dm'): self.dm = MLMD_Trajectory_DataSet(self.hparams, self.traj.traj.unsqueeze(0))

		mae = self.mlmd.model.criterion(pred, target, forecasting=True)
		one_cycle = int(2*np.pi/self.hparams.dt)
		max_mae = (target[:one_cycle]-target[one_cycle//2:int(one_cycle+one_cycle//2)]).abs().mean()

		fig_seq_pred = self.dm.plot_prediction(y=target, pred=pred, extra_title=f" {self.hparams.model_nicestr} Mode: {self.hparams.integration_mode}",
										extra_text=f'Training \n '
											 f'T: {self.t/self.hparams.dt} ({int(self.hparams.val_split* self.hparams.init_T*self.hparams.dt / (2*np.pi)):.1f}) \n '
											 f'dt: {self.hparams.dt:.3f} \n'
											 f'Epochs: {int(self.mlmd.trainer.current_epoch)} \n'
											 f'MAE: {self.mlmd.val_loss:.3f}'
											 f'\nTest \n'
											 f'T: {target.shape[0]} \n'
											 f'MAE: {mae:.3f} \n'
												f'MaxMAE: {1.27}')

		# fig_spectrum_pred = self.dm.plot_spectrum(y=target, pred=pred)

		return {'Val Forecast': fig_seq_pred}

	def forecast(self):
		'''
		The main function that performs the forecasting and calls the check_mlmd_forecast
		'''

		self.collect_init_traj()
		# self.plot_traj_prediction()
		# exit()
		self.train_mlmd()

		last_check = 0
		while len(self.traj) <= self.hparams.T:

			if last_check >= self.control_every_n_steps:
				last_check = 0

				while self.retrain_mlmd_required() :
					self.mode('simmd')
					self.simmd_forecast(int(3*self.control_every_n_steps), cat=True) # forecasting the trajectory with SimMD
					self.train_mlmd()
				self.mode('mlmd')
			else:
				last_check += self.traj.traj_segment_lengths[-1].item()

			self.mlmd_forecast(self.control_every_n_steps, cat=True)

		self.plot_traj_prediction()

	def iterative_forecast(self):

		self.collect_init_traj()

		data = []
		val_losses = []

		while len(self.traj) <= (2*2*np.pi/self.hparams.dt):
			self.train_mlmd()
			val_losses.append(self.mlmd.val_loss)
			data.append(len(self.traj))
			self.simmd_forecast(T=10, cat=True)

			steps = int(8 * 2 * np.pi / self.hparams.dt)
			target = self.simmd(T=steps, x0=self.traj.traj[-1:], t0=self.t)
			pred_x_init = target[:1] if 'lstm' in self.hparams.model else target[:1]
			pred = self.mlmd.model.forecast(T=steps, x0=pred_x_init, t0=self.t)

			target.squeeze_(0)
			pred.squeeze_(0)

			mae = self.mlmd.model.criterion(pred, target, forecasting=True)
			print(f"Step: {len(self.traj)}: MAE: {mae:.4f}")
			fig_seq_pred, fig_spectrum_pred = self.plot_traj_prediction()
			del fig_spectrum_pred
			plt.show()

		plt.plot(data, val_losses)
		plt.title(f"{self.hparams.model_nicestr}")
		plt.xlabel(f'Steps (1 Cycle={int(2*np.pi/self.hparams.dt)} steps)')
		plt.ylabel('MAE')
		plt.show()

		# self.plot_traj_prediction()




hparams = Forecasting_HParamParser(	logger=False, show=True, load_pretrained=False, save_weights=False, fast_dev_run=False, plot=True,
					project='forecasting',
					dataset='cosine',
				     	model='bi_lstm', interpolation='adiabatic', integration_mode='int',
					num_layers=2, num_hidden_multiplier=10,
				     	init_T=300, dt=(2*np.pi/100), cosine_freq=1, T=2000,
				     	input_length=1, output_length_train=10, output_length_val=10, batch_size=100,
				     	train_traj_repetition=5000, max_epochs=100, val_split=0.8)



hparams.__dict__.update({'in_features': 2})
hparams.__dict__.update({'num_hidden': hparams.in_features * hparams.num_hidden_multiplier})

if hparams.logger is True:
	os.system('wandb login --relogin afc4755e33dfa171a8419620e141ebeaeb8f27f5')
	logger = WandbLogger(project=hparams.project, entity='mlmd', name=hparams.experiment)
	hparams.__dict__.update({'logger': logger})

mlmd = MLMD(hparams)

mlmd.collect_init_traj()
mlmd.train_mlmd()
# mlmd.simmd_forecast(45)
# mlmd.simmd_forecast(128)
# mlmd.simmd_forecast(45)
# mlmd.traj.plot_simmd_traj_segments()
# mlmd.plot_traj_prediction()

# mlmd.iterative_forecast()

# traj = mlmd.traj.traj
# energy = traj.pow(2).sum(dim=-1)
# print(f"{energy=}")
# plt.plot(traj.pow(2).sum(dim=-1))
# plt.ylim(0,2)
# plt.show()
