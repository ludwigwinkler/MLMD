import sys, os
import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader

fontsize = 20
params = {'legend.fontsize': fontsize,
	  ''
	  # 'legend.handlelength': 2,
	  # 'text.usetex': True}

	  'xtick.labelsize': fontsize,
	  'ytick.labelsize': fontsize,
	  'axes.labelsize': fontsize,
	  }
plt.rcParams.update(params)

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=5, sci_mode=False)
np.set_printoptions(precision=5, suppress=True)

sys.path.append("../../..")  # Up to -> KFAC -> Optimization -> PHD
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

cwd = os.path.abspath(os.getcwd())
os.chdir(cwd)

from DiffEqNets.MolecularDynamics.src.MD_Geometry import Atom

from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning import Trainer, seed_everything
import wandb

# seed_everything(0)

from DiffEqNets.MolecularDynamics.src.MD_Models import MD_ODE, MD_Hamiltonian, MD_RNN, MD_LSTM
from DiffEqNets.MolecularDynamics.src.MD_Models import MD_BiDirectional_RNN, MD_BiDirectional_Hamiltonian, MD_BiDirectional_ODE, MD_BiDirectional_LSTM
from DiffEqNets.MolecularDynamics.src.MD_HyperparameterParser import HParamParser
from DiffEqNets.MolecularDynamics.src.MD_DataUtils import load_data, load_dm_data, QuantumMachine_DFT, Sequential_TimeSeries_DataSet, \
	Sequential_BiDirectional_TimeSeries_DataSet


class Model(LightningModule):

	def __init__(self, scaling, **kwargs):
		super().__init__()
		self.save_hyperparameters()

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

		del self.hparams['scaling']  # Can't store np.arrays during logging, so delete some after initialization

	def forward(self, t, x):

		return self.model(t, x)

	def training_step(self, batch, batch_idx):

		batch_y0, batch_t, batch_y = batch
		batch_pred = self.forward(batch_t[0], batch_y0)

		# batch_loss = F.mse_loss(batch_pred, batch_y)
		# batch_loss = F.l1_loss(batch_pred, batch_y)
		# print(batch_pred.shape)
		# exit()
		batch_loss = (batch_pred - batch_y).abs().sum(dim=1).mean()

		tqdm_dict = {'t': batch_t[0]}
		logs = {'train/mse': batch_loss}
		output = {'loss': batch_loss, 'progress_bar': tqdm_dict, 'log': logs}
		return output

	def validation_step(self, batch, batch_idx):
		batch_y0, batch_t, batch_y = batch

		batch_pred = self.forward(batch_t[0], batch_y0)

		# print(f"{batch_pred.shape=} {batch_y.shape=}")

		batch_loss = F.mse_loss(batch_pred, batch_y)

		tqdm_dict = {'t': batch_t[0]}
		output = {'val_loss': batch_loss}
		if self.hparams.data_set in ['benzene_dft.npz', 'ethanol_dft.npz', 'malonaldehyde_dft.npz']:
			batch_pred_angles, batch_pred_distances = Atom(batch_pred, hparams).compute_MD_geometry()
			batch_angles, batch_distances = Atom(batch_y, hparams).compute_MD_geometry()

			batch_angle_loss = F.mse_loss(batch_pred_angles, batch_angles)
			batch_distance_loss = F.mse_loss(batch_pred_distances, batch_distances)

			tqdm_dict.update({'Angle': batch_angle_loss, 'Dist': batch_distance_loss})
			output.update({'Angle': batch_angle_loss, 'Dist': batch_distance_loss})

		output.update({'progress_bar': tqdm})
		return output

	def validation_epoch_end(self, outputs):
		loss = torch.stack([x['val_loss'] for x in outputs]).mean()
		logs = {'val/mse': loss}
		if False and self.hparams.data_set in ['benzene_dft.npz', 'ethanol_dft.npz', 'malonaldehyde_dft.npz']:
			angle = torch.stack([x['Angle'] for x in outputs]).mean()
			dist = torch.stack([x['Dist'] for x in outputs]).mean()
			# self.plot_velocity_histogram()
			# self.plot_predictions()
			tqdm_dict = {'val_loss': loss, 'Angle': angle, 'Dist': dist}
			logs.update({'val/angle': angle, 'val/dist': dist})
		else:
			tqdm_dict = {'val_loss': loss}

		return {'val_loss': loss, 'val/mse': loss, 'progress_bar': tqdm_dict, 'log': logs}

	def training_epoch_end(self, outputs):
		loss = torch.stack([x['loss'] for x in outputs]).mean()
		logs = {'train/mse': loss}
		if False and self.hparams.data_set in ['benzene_dft.npz', 'ethanol_dft.npz', 'malonaldehyde_dft.npz']:
			angle = torch.stack([x['Angle'] for x in outputs]).mean()
			dist = torch.stack([x['Dist'] for x in outputs]).mean()
			# self.plot_velocity_histogram()
			# self.plot_predictions()
			tqdm_dict = {'loss': loss, 'Angle': angle, 'Dist': dist}
			logs.update({'train/angle': angle, 'train/dist': dist})
		else:
			tqdm_dict = {'train_loss': loss}

		return {'loss': loss, 'train/mse': loss, 'progress_bar': tqdm_dict, 'log': logs}

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
		if self.hparams.data_set in ['benzene_dft.npz', 'ethanol_dft.npz', 'malonaldehyde_dft.npz']:
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
		# exit()

	@torch.no_grad()
	def predict_sequentially(self, dm=None):
		if dm is None:
			data_train = self.trainer.datamodule.val_dataloader().dataset.data
		elif dm is not None:
			data_train = dm.val_dataloader().dataset.data
		assert data_train.dim() == 3
		num_sequential_samples = 1000
		if 'bi' in self.hparams.model:
			data_set = Sequential_BiDirectional_TimeSeries_DataSet(data_train, input_length=self.hparams.input_length,
									       output_length=self.hparams.output_length)
			dataloader = DataLoader(data_set, batch_size=11,
						sampler=torch.utils.data.SequentialSampler(data_set.data))
			t_y0 = []
			segments = num_sequential_samples // (2 * self.hparams.input_length + self.hparams.output_length)
			input_length_ = np.tile(np.arange(0, self.hparams.input_length), segments)

			# print(input_length_)
			segment_offset = np.repeat(np.arange(0, segments) * (self.hparams.input_length + self.hparams.output_length),self.hparams.input_length)
			if self.hparams.input_length == 1:  # I don't know why played around and it finally looked ok
				segment_offset = np.repeat(np.arange(0, segments) * (self.hparams.input_length + self.hparams.output_length-1),self.hparams.input_length)
			t_y0 = input_length_ + segment_offset

		else:
			data_set = Sequential_TimeSeries_DataSet(data_train, input_length=self.hparams.input_length,
								 output_length=self.hparams.output_length)
			dataloader = DataLoader(data_set, batch_size=11,
						sampler=torch.utils.data.SequentialSampler(data_set.data))
			segments = num_sequential_samples // (self.hparams.input_length + self.hparams.output_length)

			input_length_ = np.tile(np.arange(0, self.hparams.input_length), segments)
			segment_offset = np.repeat(np.arange(0, segments) * (self.hparams.input_length + self.hparams.output_length),self.hparams.input_length)
			if self.hparams.input_length == 1: # I don't know why played around and it finally looked ok
				segment_offset = np.repeat(np.arange(0, segments) * (self.hparams.input_length + self.hparams.output_length-1),self.hparams.input_length)
			t_y0 = input_length_ + segment_offset

		y_ = []
		y0_ = []
		pred_ = []
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
			y0_.append(y0[:, :(self.hparams.input_length)].flatten(0, 1))
		pred = torch.cat(pred_, dim=0).cpu().detach().numpy()
		ys = torch.cat(y_, dim=0).cpu().detach().numpy()
		y0s = torch.cat(y0_, dim=0).cpu().detach().numpy()

		# ys = ys*dm.data_std + dm.data_mean
		# pred = pred*dm.data_std + dm.data_mean
		# y0s = y0s*dm.data_std + dm.data_mean

		# print(ys.mean(dim=[0])-dm.data_mean)

		# plt.title('animation')
		# plt.hist(ys[:,:,ys.shape[-1]//2], bins=100, density=True)
		# plt.show()

		torch.save(pred, 'AnimationPred.pt')
		torch.save(ys, 'AnimationTrue.pt')
		torch.save(y0s, 'AnimationConditions.pt')

		colors=['r', 'g', 'b']

		plt.plot(ys[:num_sequential_samples, 0], color=colors[0], label='Data')
		plt.plot(ys[:num_sequential_samples, 1], color=colors[1])
		plt.plot(ys[:num_sequential_samples, 2], color=colors[2])
		plt.plot(pred[:num_sequential_samples, 0], ls='--', color=colors[0], label='Prediction')
		plt.plot(pred[:num_sequential_samples, 1], ls='--', color=colors[1])
		plt.plot(pred[:num_sequential_samples, 2], ls='--', color=colors[2])

		plt.scatter(t_y0, y0s[:t_y0.shape[0], 0], color=colors[0], label='Initial and Final Conditions')
		plt.scatter(t_y0, y0s[:t_y0.shape[0], 1], color=colors[1], label='Initial and Final Conditions')
		plt.scatter(t_y0, y0s[:t_y0.shape[0], 2], color=colors[2], label='Initial and Final Conditions')
		plt.xlabel('t')
		plt.ylabel('$q(t)$')
		plt.title(self.hparams.data_set)
		plt.legend()
		plt.show()

	def on_fit_end(self):

		if self.hparams.plot:
			self.predict_sequentially()


hparams = HParamParser(logname='MD', logger=False, plot=True,
		       model='lstm',
		       data_set='keto_1fs.npz', input_length=3, output_length=20,
		       train_traj_repetition=1, pct_data_set=0.001)
hparams.__dict__.update({'logname': hparams.logname + '_pct' + str(hparams.pct_data_set) + '_' + str(hparams.model) + '_' + str(hparams.data_set) + '_T' + str(hparams.output_length)})

dm = load_dm_data(hparams)
print(dm)

exit()

scaling = {'y_mu': dm.y_mu, 'y_std': dm.y_std, 'dy_mu': dm.dy_mu, 'dy_std': dm.y_std}
hparams.__dict__.update({'in_features': dm.y_mu.shape[-1]})
hparams.__dict__.update({'num_hidden': dm.y_mu.shape[-1]*5})

model = Model(scaling, **vars(hparams))

if hparams.logger is True:
	# logger = TensorBoardLogger(save_dir=hparams.logdir, name=hparams.logname)
	from pytorch_lightning.loggers import WandbLogger

	# os.environ['WANDB_API_KEY'] = 'afc4755e33dfa171a8419620e141ebeaeb8f27f5'
	os.system('wandb login --relogin afc4755e33dfa171a8419620e141ebeaeb8f27f5')
	# wandb.login(key='afc4755e33dfa171a8419620e141ebeaeb8f27f5', relogin=True)
	logger = WandbLogger(project='BiDirectional_MD', entity='mlmd',
			     name=hparams.logname)
	hparams.__dict__.update({'logger': logger})

early_stop_callback = EarlyStopping(
	monitor='val/mse',
	min_delta=0.00,
	patience=3,
	verbose=True,
	mode='min',

)

# most basic trainer, uses good defaults
trainer_ = Trainer()
trainer = Trainer.from_argparse_args(hparams,
				     max_epochs=2000,
				     min_steps=5000,
				     # max_steps=50,
				     progress_bar_refresh_rate=5,
				     early_stop_callback=early_stop_callback,
				     # limit_train_batches=10,
				     # limit_val_batches=10,
					val_check_interval=1.,
				     fast_dev_run=False,
				     checkpoint_callback=False,
				     gpus=hparams.gpus
				     )
# checkpoint = torch.load('Animation.ckpt')

# trainer.on_save_checkpoint('checkpoints.epoch=14_v0.ckpt')
# model = Model.load_from_checkpoint(scaling=scaling,checkpoint_path='checkpoints/epoch=14_v0.ckpt')
trainer.fit(model, datamodule=dm)

# model.predict_sequentially(dm)
