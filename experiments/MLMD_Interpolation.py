import math
import sys, os, inspect, copy, time

from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.patches import Patch
from matplotlib.lines import Line2D
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes, mark_inset


import torch


from torchtyping import TensorType, patch_typeguard
from typeguard import typechecked

patch_typeguard()

fontsize = 60
params = {
 'font.size'       : fontsize,
 'legend.fontsize' : fontsize * 0.75,
 'xtick.labelsize' : fontsize,
 'ytick.labelsize' : fontsize,
 'axes.labelsize'  : fontsize,
 'figure.figsize'  : (12, 12),
 'figure.facecolor': 'white',
 'lines.linewidth' : 3,
 'text.usetex'     : True,
 'mathtext.fontset': 'stix',
 'font.family'     : 'STIXGeneral'}
plt.rcParams.update(params)

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor



# This is a test

torch.set_printoptions(precision=5, sci_mode=False)
np.set_printoptions(precision=5, suppress=True)

sys.path.append("/".join(os.getcwd().split("/")[:-3])) # experiments -> MLMD -> PHD
sys.path.append("/".join(os.getcwd().split("/")[:-2])) # experiments -> MLMD
sys.path.append("/".join(os.getcwd().split("/")[:-1])) # experiments
sys.path.append(os.getcwd())

file_path = Path(__file__)
cwd = file_path.parent

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

import pytorch_lightning
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping, ModelCheckpoint
from pytorch_lightning import Trainer, seed_everything

import wandb
from pytorch_lightning.loggers import WandbLogger

# seed_everything(123)

import MLMD

from MLMD.src.MD_PL_CallBacks import OverwritingModelCheckpoint
from MLMD.src.MD_ModelUtils import auto_scale_batch_size
from MLMD.src.MD_HyperparameterParser import Interpolation_HParamParser
from MLMD.src.MD_DataUtils import load_dm_data, QuantumMachine_DFT, Sequential_TimeSeries_DataSet, \
	Sequential_BiDirectional_TimeSeries_DataSet
from MLMD.src.MD_Utils import Benchmark, Timer
from MLMD.src.MD_PLModules import Interpolator

plotting_mode = ['vibrationalspectra', 'interatomicdistances', 'aldehydeplot', 'training'][0]

if plotting_mode=='training':

	hparams = Interpolation_HParamParser(logger=0, plot=0, show=0, load_weights=0, save_weights=0, fast_dev_run=0,
										 project='vibrationalspectra',
										 model='bi_lstm', num_layers=5, num_hidden_multiplier=10, criterion='MAE',
										 interpolation=True, interpolation_mode='adiabatic', integration_mode='diffeq',
										 diffeq_output_scaling=1,
										 dataset=['malonaldehyde_dft.npz', 'benzene_dft.npz', 'ethanol_dft.npz',
												  'toluene_dft.npz', 'naphthalene_dft.npz', 'salicylic_dft.npz',
												  'paracetamol_dft.npz', 'aspirin_dft.npz',
												  'keto_100K_0.2fs.npz', 'keto_300K_0.2fs.npz', 'keto_500K_0.2fs.npz'
												  ][0],
										 input_length=1, output_length=20, batch_size=49, auto_scale_batch_size=False,
										 optim='adam', lr=1e-3, train_traj_repetition=1, max_epochs=2,
										 limit_train_batches=25, limit_val_batches=25)

	dm = load_dm_data(hparams)

	if hparams.auto_scale_batch_size:
		new_batch_size = auto_scale_batch_size(hparams, Interpolator, dm)
		hparams.__dict__.update({'batch_size': new_batch_size})

	model = Interpolator(**vars(hparams))
	model.model.set_diffeq_output_scaling_statistics(dm.dy_mu, dm.dy_std)

	if hparams.logger:
		os.system('wandb login --relogin afc4755e33dfa171a8419620e141ebeaeb8f27f5')
		logger = WandbLogger(project=hparams.project, entity='mlmd', name=hparams.experiment)
		hparams.__dict__.update({'logger': logger})

	early_stop_callback = EarlyStopping(monitor='Val/Epoch' + hparams.criterion, mode='min',
										patience=3, min_delta=0.0005,
										verbose=True)

	checkpoint_callback = OverwritingModelCheckpoint(monitor='Val/Epoch' + hparams.criterion,
													 dirpath=f'{cwd}/ckpt/',
													 filename=hparams.ckptname,
													 save_weights_only=True,
													 mode='min')

	trainer = Trainer.from_argparse_args(hparams,
										 # min_steps=1000,
										 # max_steps=50,
										 progress_bar_refresh_rate=5,
										 callbacks=[early_stop_callback,
													checkpoint_callback] if hparams.save_weights else [
											 early_stop_callback],
										 val_check_interval=1.,
										 gpus=1 if torch.cuda.is_available() else None,
										 distributed_backend=None
										 )

	trainer.fit(model=model, dm=dm)

if plotting_mode=='vibrationalspectra':
	
	fontsize = 60
	params = {
	 'font.size'       : fontsize,
	 'legend.fontsize' : fontsize * 0.75,
	 'xtick.labelsize' : fontsize,
	 'ytick.labelsize' : fontsize,
	 'axes.labelsize'  : fontsize,
	 'figure.figsize'  : (12, 12),
	 'figure.facecolor': 'white',
	 'lines.linewidth' : 3,
	 'text.usetex'     : True,
	 'mathtext.fontset': 'stix',
	 'font.family'     : 'STIXGeneral'}
	plt.rcParams.update(params)

	fig, axs = plt.subplots(ncols=1, nrows=6, figsize=(20, 54), gridspec_kw={'height_ratios': [2, 1, 2, 1, 2, 1]})
	max_diff = 0

	for T in [5, 20, 80]:

		i = {5:0, 20:2, 80:4}[T]
		print()
		print(f"{T=}")
		print()

		trained_spectra_dicts = []
		untrained_spectra_dicts = []

		hparams = Interpolation_HParamParser(	logger=0, plot=0, show=0, load_weights=0, save_weights=0, fast_dev_run=0,
							project='vibrationalspectra',
							model='bi_lstm', num_layers=5, num_hidden_multiplier=10, criterion='MAE',
							interpolation=True, interpolation_mode='adiabatic', integration_mode='diffeq', diffeq_output_scaling=1,
							dataset=['malonaldehyde_dft.npz', 'benzene_dft.npz', 'ethanol_dft.npz', 'toluene_dft.npz', 'naphthalene_dft.npz', 'salicylic_dft.npz', 'paracetamol_dft.npz', 'aspirin_dft.npz',
								 'keto_100K_0.2fs.npz', 'keto_300K_0.2fs.npz', 'keto_500K_0.2fs.npz'
								 ][0],
							input_length=1, output_length=T, batch_size=49, auto_scale_batch_size=False,
							optim='adam', lr=1e-3, train_traj_repetition=1, max_epochs=2,
							limit_train_batches=25, limit_val_batches=25)

		dm = load_dm_data(hparams)

		model = Interpolator(**vars(hparams))
		model.model.set_diffeq_output_scaling_statistics(dm.dy_mu, dm.dy_std)

		num_sequential_samples = 100000
		for TRAINED in [False, True]:
			model = Interpolator(**vars(hparams))
			model.model.set_diffeq_output_scaling_statistics(dm.dy_mu, dm.dy_std)
			if TRAINED: model.load_weights()

			if os.path.exists((f"{cwd}/MDPredictions/Trained" if TRAINED else f"{cwd}/MDPredictions/Untrained")+f"/VibSpectra_{hparams.ckptname}_N{num_sequential_samples}"):
				vibspectra_dict = torch.load((f"{cwd}/MDPredictions/Trained" if TRAINED else f"{cwd}/MDPredictions/Untrained") + f"/VibSpectra_{hparams.ckptname}_N{num_sequential_samples}")
				print(f"Vibspectra loaded from disk ...")

			else:
				y, pred, y0, t0 = model.predict_sequentially(dm, num_sequential_samples=num_sequential_samples)

				y = torch.cat(y)
				pred = torch.cat(pred)
				y = dm.unnormalize(y)
				pred = dm.unnormalize(pred)
				y0 = dm.unnormalize(y0)

				# dm.save_as_npz(pred, y, y0, name=hparams.ckptname, path=f"{cwd}/MDPredictions/Trained" if TRAINED else f"{cwd}/MDPredictions/Untrained")
				vibspectra_dict = dm.plot_vibrational_spectra(y=y, pred=pred, extra_title=f'Interpolation {hparams.dataset_nicestr}, T={hparams.output_length_val}', show=False)
				torch.save(obj=vibspectra_dict, f=(f"{cwd}/MDPredictions/Trained" if TRAINED else f"{cwd}/MDPredictions/Untrained") + f"/VibSpectra_{hparams.ckptname}_N{num_sequential_samples}")


			if TRAINED: trained_spectra_dicts += [vibspectra_dict]
			else: untrained_spectra_dicts += [vibspectra_dict]

		# for i, (T, dict) in enumerate(zip([5, 10, 20, 40, 80], untrained_spectra_dicts)):
		# 	plt.plot(dict['frequency'], dict['pred_pdos'], label=f'Interpolation T={T}', ls='--')

		assert np.sum(np.abs((trained_spectra_dicts[-1]['pred_pdos'] - trained_spectra_dicts[-1]['true_pdos'])))>0.0
		# print(f"{np.sum(np.abs((trained_spectra_dicts[-1]['pred_pdos'] - trained_spectra_dicts[-1]['true_pdos'])))=}")
		scaling = untrained_spectra_dicts[-1]['true_pdos'][0] / untrained_spectra_dicts[-1]['pred_pdos'][0] if T!=80 else 1
		untrained_spectra_dicts[-1]['pred_pdos'] = untrained_spectra_dicts[-1]['pred_pdos'] * scaling
		
		diff = trained_spectra_dicts[-1]['pred_pdos'] - trained_spectra_dicts[-1]['true_pdos']
		max_diff = np.max(np.abs(diff)) if np.max(np.abs(diff)) > max_diff else max_diff
		

		
		axs[i].plot(trained_spectra_dicts[-1]['frequency'], trained_spectra_dicts[-1]['true_pdos'], 	label='Ground Truth' if T==5 else None, alpha=0.5, color='orange')
		axs[i].plot(trained_spectra_dicts[-1]['frequency'], trained_spectra_dicts[-1]['pred_pdos'], 	label=f'MLMD T={T} fs', alpha=0.5, color='blue')
		axs[i].plot(untrained_spectra_dicts[-1]['frequency'], untrained_spectra_dicts[-1]['pred_pdos'], label=f'Interpolation T={T} fs', alpha=0.5, color='red')
		axs[i].fill_between(untrained_spectra_dicts[-1]['frequency'], 0, untrained_spectra_dicts[-1]['pred_pdos'], alpha=0.25, color='red')

		axs[i+1].plot(trained_spectra_dicts[-1]['frequency'], diff, color='blue', alpha=0.5)
		axs[i+1].plot(trained_spectra_dicts[-1]['frequency'], np.zeros_like(diff), color='black', alpha=0.5)
		axs[i+1].ticklabel_format(axis='y', style='sci', scilimits=(-1, 1), useMathText=True)
		axs[i+1].set_ylim(-4*10**-4, 4 * 10 ** -4)

		if T==80:
			maxerror_index = np.argmax(np.abs(diff))
			maxerror_index = 5500
			print(f"{maxerror_index=}")
			axins = zoomed_inset_axes(parent_axes=axs[i], zoom=15, loc='upper right')
			# axins.set_ylim(untrained_spectra_dicts[-1]['true_pdos'][maxerror_index] - 0.005, untrained_spectra_dicts[-1]['true_pdos'][maxerror_index] + 0.005)
			# axins.plot(untrained_spectra_dicts[-1]['frequency'][:100], untrained_spectra_dicts[-1]['pred_pdos'][:100],label=f'Interpolation T={T}', alpha=0.5, color='black')
			index_start = 5430
			index_end = 5550
			axins.plot(trained_spectra_dicts[-1]['frequency'][index_start:index_end], trained_spectra_dicts[-1]['true_pdos'][index_start: index_end],alpha=0.5, color='orange')
			axins.plot(trained_spectra_dicts[-1]['frequency'][index_start:index_end], trained_spectra_dicts[-1]['pred_pdos'][index_start: index_end],alpha=0.5, color='blue')
			axins.set_yticks([])
			axins.set_xticks([])
			axins.set_title('15x')
			mark_inset(axs[i], axins, loc1=2, loc2=4, fc="none", ec="0.5")

		# axs[i].legend(frameon=False, loc='upper center')
			'''Custom legend'''
		legend_elements = [Line2D([0], [0], color='orange', lw=4, label='Ground Truth'),
		                   Line2D([0], [0], color='blue', lw=4, label='MLMD Prediction'),
		                   Patch(facecolor='red', alpha=0.5, edgecolor='red', label='From Training Data')]
		axs[0].legend(frameon=False, handles=legend_elements, loc='upper right')
		axs[i].set(frame_on=False)
		axs[i].set_xlim(0, 3200) # Frequency cut off
		axs[i+1].set_xlim(0, 3200) # Frequency cut off
		
		axs[i].set_ylabel('VDOS [a.u.]')
		axs[i].set_xlabel('Frequency [cm$^{-1}$]')
		
		axs[i+1].set(frame_on=False)
		axs[i+1].set_ylabel('Difference [a.u.]')
		axs[i+1].set_xlabel('Frequency [cm$^{-1}$]')

		# axs[i].set_title(f'Velocity Auto-Correlation Function of Trained Models' if TRAINED else f'Velocity Auto-Correlation Function of Interpolation')

	plt.tight_layout()
	plt.savefig(f"VelAutoCorrFuncTrainedCombined.pdf" if TRAINED else f"VelAutoCorrFuncUntrainedCombined.pdf")
	plt.savefig(f"VelAutoCorrFuncTrainedCombined.svg" if TRAINED else f"VelAutoCorrFuncUntrainedCombined.svg")
	plt.show()

if plotting_mode=='interatomicdistances':

	fontsize = 60
	params = {'font.size': fontsize,
			  'legend.fontsize': fontsize*0.75,
			  'xtick.labelsize': fontsize,
			  'ytick.labelsize': fontsize,
			  'axes.labelsize': fontsize,
			  'figure.figsize': (12, 12),
			  'figure.facecolor': 'white',
			  'lines.linewidth': 3,
			  'text.usetex': True,
			  'mathtext.fontset': 'stix',
			  'font.family': 'STIXGeneral'
			  }
	plt.rcParams.update(params)
	
	for DATASET_ITERATOR in [-1]:
		hparams = Interpolation_HParamParser(logger=0, plot=0, show=0, load_weights=1, save_weights=0, fast_dev_run=0,
											 project='vibrationalspectra',
											 model='bi_lstm', num_layers=5, num_hidden_multiplier=10, criterion='MAE',
											 interpolation=True, interpolation_mode='adiabatic', integration_mode='diffeq',
											 diffeq_output_scaling=1,
											 dataset=['malonaldehyde_dft.npz', 'benzene_dft.npz', 'ethanol_dft.npz',
													  'toluene_dft.npz', 'naphthalene_dft.npz', 'salicylic_dft.npz',
													  'paracetamol_dft.npz', 'aspirin_dft.npz',
													  'keto_100K_0.2fs.npz', 'keto_300K_0.2fs.npz', 'keto_500K_0.2fs.npz'
													  ][DATASET_ITERATOR],
											 input_length=1, output_length=20, batch_size=200 if torch.cuda.is_available() else 64, auto_scale_batch_size=False,
											 optim='adam', lr=1e-3, train_traj_repetition=1, max_epochs=2000 if torch.cuda.is_available() else 2
											 )
	
		dm = load_dm_data(hparams)
	
		if hparams.auto_scale_batch_size:
			new_batch_size = auto_scale_batch_size(hparams, Interpolator, dm)
			hparams.__dict__.update({'batch_size': new_batch_size})
	
		model = Interpolator(**vars(hparams))
		model.model.set_diffeq_output_scaling_statistics(dm.dy_mu, dm.dy_std)
	
		model.load_weights()
		model.plot_interatomic_distances_histogram(dm)
		model.plot_speed_histogram(dm)

if plotting_mode=='aldehydeplot':
	
	print(f"\n Aldehyde plot \n ")
	
	for DATASET_ITERATOR in [-1]:
		hparams = Interpolation_HParamParser(logger=1, plot=0, show=0, load_weights=1, save_weights=0, fast_dev_run=0,
											 project='vibrationalspectra',
											 model='bi_lstm', num_layers=5, num_hidden_multiplier=10, criterion='MAE',
											 interpolation=True, interpolation_mode='adiabatic', integration_mode='diffeq',
											 diffeq_output_scaling=1,
											 dataset=['malonaldehyde_dft.npz', 'benzene_dft.npz', 'ethanol_dft.npz',
													  'toluene_dft.npz', 'naphthalene_dft.npz', 'salicylic_dft.npz',
													  'paracetamol_dft.npz', 'aspirin_dft.npz',
													  'keto_100K_0.2fs.npz', 'keto_300K_0.2fs.npz', 'keto_500K_0.2fs.npz'
													  ][DATASET_ITERATOR],
											 input_length=1, output_length=5,
											 batch_size=200 if torch.cuda.is_available() else 64,
											 auto_scale_batch_size=False,
											 optim='adam', lr=1e-3, train_traj_repetition=1,
											 max_epochs=2000 if torch.cuda.is_available() else 2
											 )
	
		dm = load_dm_data(hparams)
		model = Interpolator(**vars(hparams))
		model.model.set_diffeq_output_scaling_statistics(dm.dy_mu, dm.dy_std)
	
	
		TRAINED = True
		if TRAINED: model.load_weights()
		# y, pred, y0, t0 = model.predict_sequentially(dm)
		y, pred, y0, t0 = model.predict_sequentially(dm, num_sequential_samples=int(dm.data.shape[1]*0.88))
		# y, pred, y0, t0 = model.predict_sequentially(dm, num_sequential_samples=2000)
		# y, pred, y0, t0 = model.predict_sequentially(dm, num_sequential_samples=300000)
	
		y = torch.cat(y)
		pred = torch.cat(pred)
		y = dm.unnormalize(y)
		pred = dm.unnormalize(pred)
		y0 = dm.unnormalize(y0)
	
		dm.save_as_npz(pred=pred, y=y, conditions=y0, name=hparams.ckptname,
					   path=f"{cwd}/MDPredictions/Trained" if TRAINED else f"{cwd}/MDPredictions/Untrained")
	
		cwd_ = str(cwd).replace(' ', '\ ')
		# os.system(f"python {cwd_}/CorrMap2Var_ASE_v10_230620.py {cwd_}/MDPredictions/{'Trained' if TRAINED else 'Untrained'}/Pred_{hparams.ckptname}.npz dihedral[3,2,1,0] dihedral[4,0,1,2] ")
		os.system(f"python {cwd_}/CorrMap2Var_ASE_v10_230620.py {cwd_}/MDPredictions/{'Trained' if TRAINED else 'Untrained'}/True_{hparams.ckptname}.npz dihedral[3,2,1,0] dihedral[4,0,1,2] ")

# exit()

# print(vibspectra_dict['frequency_loss'])
# hparams.logger.experiment.log({'Pred': wandb.Image(fig, caption="Val Prediction")})
# hparams.logger.log_metrics({'VibSpectraLoss_MSE': vibspectra_dict['frequency_diff'].pow(2).mean(),
# 							'VibSpectraLoss_MAE': vibspectra_dict['frequency_diff'].pow(2).sum()}, step=0)

# exit()

# dm.save_as_npz(pred, y, y0, name=hparams.ckptname, path=f"{cwd}/MDPredictions")

