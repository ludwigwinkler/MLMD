import argparse
import datetime
import os, sys, shutil
sys.path.append("../..")

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm, trange
import logging

matplotlib.rcParams["figure.figsize"] = [10, 10]

if sys.platform=='linux': matplotlib.use('TKAgg')

import torch
from torch.nn import Sequential, Linear, BatchNorm1d
from torch.nn import Tanh
from torch.optim import Adam, RMSprop, SGD
from torch.utils.data import DataLoader
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

sys.path.append(os.path.abspath('../../../..'))  # .experiments -> MD -> DiffEqNets ->PhD

from torchdiffeq import odeint_adjoint as odeint
from DiffEqNets.MolecularDynamics.src.MD_Utils import str2bool, matplotlibfigure_to_tensor, clean_hparam_directory, NormalizedLoss
from DiffEqNets.MolecularDynamics.src.MD_DataUtils import load_data, plot_MD_data, plot_HMC_data, plot_Lorenz_data
from DiffEqNets.MolecularDynamics.src.MD_Models import MD_ODENet, MD_ODE2Net, MD_HamiltonianNet, MD_UniversalDiffEq
from DiffEqNets.MolecularDynamics.src.MD_Geometry import Atom
from DiffEqNets.MolecularDynamics.src._MD_Hyperparameters import HParams

from Utils.Utils import RunningAverageMeter

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

FloatTensor = torch.cuda.FloatTensor
Tensor = torch.cuda.FloatTensor

def train(model, dataloader, hparams):

	if hparams.log:
		logging.basicConfig(format='',
				    level=logging.INFO,
				    filename=f'{hparams.logname}.txt',
				    filemode='w')

	train_loader, val_loader = dataloader

	train_loss          = RunningAverageMeter(momentum=0.95)
	train_angle_loss    = RunningAverageMeter(momentum=0.95)
	train_distance_loss = RunningAverageMeter(momentum=0.95)

	val_loss            = RunningAverageMeter(momentum=0.6)
	val_angle_loss      = RunningAverageMeter(momentum=0.6)
	val_distance_loss   = RunningAverageMeter(momentum=0.6)

	if next(model.parameters()).is_cuda: print('Training on GPU')

	if hparams.verbose: epochs = range(hparams.num_epochs)
	elif not hparams.verbose: epochs = trange(hparams.num_epochs)
	for epoch in epochs:

		if epoch>1:
			train_loader.dataset.update_output_length_samplerange(low=2, high=2, mode='add')
			# train_loader.dataset.sample_output_length()
			pass

		if hparams.verbose:
			progress = tqdm(train_loader, bar_format='{l_bar}{r_bar}', total=len(train_loader)+len(val_loader))
		elif not hparams.verbose:
			progress  = train_loader
		for batch_i, (batch_y0, batch_t, batch_y) in enumerate(progress):

			# plt.plot(batch_y[0,:,0], batch_y[0,:,1])
			# plt.show()
			# print(f"{batch_y.shape=}")
			# exit()

			# print(f'{batch_y0.shape=}')
			# print(f'{batch_t.shape=}')
			# print(f'{batch_y.shape=}')
			# exit('@MD_Train')

			model.optim.zero_grad()

			batch_y0, batch_t, batch_y = batch_y0.to(device), batch_t.to(device), batch_y.to(device)

			# print(f'@train {batch_t=} {batch_y0.shape=}')
			# exit()
			batch_pred = model(t=batch_t[0], x=batch_y0) # data three dimensional: [ timesteps, batchsize, features ]

			# print("@MD train")
			# print(f'{batch_y0.shape=} {batch_y.shape=} {batch_pred.shape=}')
			# print(batch_y0[0])
			# print(batch_y[0,0])
			# print(batch_pred[0,0])
			# exit()
			batch_loss = model.criterion(batch_pred, batch_y)

			# if hparams.log and (batch_i%50==0):
			# 	hparams.logger.add_histogram(tag='Input', values=model.net.input, global_step=train_loss.step)
			# 	hparams.logger.add_histogram(tag='Output', values=model.net.output, global_step=train_loss.step)

			batch_loss.backward()
			model.optim.step()
			train_loss.update(batch_loss.item())

			if batch_i%10==0 and epoch>=0 and batch_i>0 and True:

				if hparams.train_data in ['benzene_dft.npz',
							  'ethanol_dft.npz',
							  'malonaldehyde_dft.npz',
							  'H2O/HigherEnergy/H2O_HigherEnergy1.npz',
							  'H2O/HigherEnergy/H2O_HigherEnergy2.npz',
							  'H2O/HigherEnergy/H2O_HigherEnergy3.npz',
							  'H2O/LowerEnergy/H2O_LowerEnergy1.npz',
							  'Ethanol']:

					batch_pred_angles, batch_pred_distances 	= Atom(batch_pred, hparams).compute_MD_geometry()
					batch_angles, batch_distances 			= Atom(batch_y, hparams).compute_MD_geometry()

					# print(batch_angles)
					# print(batch_pred_angles)
					# exit()

					batch_angle_loss 	= F.l1_loss(batch_pred_angles, batch_angles)
					batch_distance_loss 	= F.l1_loss(batch_pred_distances, batch_distances)

					train_angle_loss.update(batch_angle_loss.item())
					train_distance_loss.update(batch_distance_loss.item())

			if batch_i==len(train_loader)-1:

				model.eval()

				with torch.no_grad():

					# print(f'{val_loader.dataset.output_length=} {train_loader.dataset.output_length=}')
					val_loader.dataset.output_length = train_loader.dataset.output_length
					# print(f'{val_loader.dataset.output_length=} {train_loader.dataset.output_length=}')
					# exit()

					for batch_i, (batch_val_y0, batch_val_t, batch_val_y) in enumerate(val_loader):

						batch_val_y0, batch_val_t, batch_val_y = batch_val_y0.to(device), batch_val_t.to(device), batch_val_y.to(device)
						batch_val_t = batch_val_t[0]

						batch_val_pred 	= model(batch_val_t,batch_val_y0)
						batch_val_loss 	= model.criterion(batch_val_pred, batch_val_y)

						val_loss.update(batch_val_loss.item())

						if hparams.train_data in ['benzene_dft.npz',
									  'ethanol_dft.npz',
									  'malonaldehyde_dft.npz',
									  'H2O/HigherEnergy/H2O_HigherEnergy1.npz',
									  'H2O/HigherEnergy/H2O_HigherEnergy2.npz',
									  'H2O/HigherEnergy/H2O_HigherEnergy3.npz',
									  'H2O/LowerEnergy/H2O_LowerEnergy1.npz',
									  'Ethanol']:
							batch_val_pred_angles, batch_val_pred_distances 	= Atom(batch_val_pred, hparams).compute_MD_geometry()
							batch_val_angles, batch_val_distances 			= Atom(batch_val_y, hparams).compute_MD_geometry()

							batch_val_angle_loss 	= F.l1_loss(batch_val_pred_angles, batch_val_angles)
							batch_val_distance_loss	= F.l1_loss(batch_val_pred_distances, batch_val_distances)
							val_angle_loss.update(batch_val_angle_loss.item())
							val_distance_loss.update(batch_val_distance_loss.item())
							desc = 	f'Epoch {epoch}: '\
								f'Train: Loss: {train_loss.avg:.2f} '\
								f'Angle:{train_angle_loss.avg / train_loader.dataset.output_length:.2f} '\
								f'Dist:{train_distance_loss.avg / train_loader.dataset.output_length:.2f} '\
								f't: {train_loader.dataset.output_length} '\
								f'| '\
								f'Val: Loss:{val_loss.avg:.2f} '\
								f'Angle:{val_angle_loss.avg / val_loader.dataset.output_length:.2f} '\
								f'Dist:{val_distance_loss.avg / val_loader.dataset.output_length:.2f} '\
								f't: {val_loader.dataset.output_length}'
						else:
							desc = f'Epoch {epoch} '\
								f'Train: Loss: {train_loss.avg:.2f} '\
								f't: { train_loader.dataset.output_length} '\
								f'| '\
								f'Val: Loss:{val_loss.avg:.2f}'

						if hparams.verbose:
							progress.set_description(desc)
							progress.update(1)

				model.train()

			if hparams.train_data in ['benzene_dft.npz',
						  'ethanol_dft.npz',
						  'malonaldehyde_dft.npz',
						  'H2O/HigherEnergy/H2O_HigherEnergy1.npz',
						  'H2O/HigherEnergy/H2O_HigherEnergy2.npz',
						  'H2O/HigherEnergy/H2O_HigherEnergy3.npz',
						  'H2O/LowerEnergy/H2O_LowerEnergy1.npz',
						  'Ethanol']:
				desc = f'Epoch {epoch}: ' \
				       f'Train: Loss: {train_loss.avg:.2f} ' \
				       f'Angle:{train_angle_loss.avg / train_loader.dataset.output_length:.2f} ' \
				       f'Dist:{train_distance_loss.avg / train_loader.dataset.output_length:.2f} ' \
				       f't: {train_loader.dataset.output_length} ' \
				       f'| ' \
				       f'Val: Loss:{val_loss.avg:.2f} ' \
				       f'Angle:{val_angle_loss.avg / val_loader.dataset.output_length:.2f} ' \
				       f'Dist:{val_distance_loss.avg / val_loader.dataset.output_length:.2f} ' \
				       f't: {val_loader.dataset.output_length}'
			else:
				desc = f'Epoch {epoch} ' \
				       f'Train: Loss: {train_loss.avg:.2f} ' \
				       f't: {train_loader.dataset.output_length} ' \
				       f'| ' \
				       f'Val: Loss:{val_loss.avg:.2f}'
			if hparams.verbose:
				progress.set_description(desc)
				# progress.update(1)


		# exit()

		if hparams.log:

			hparams.logger.add_scalar('Train/pred loss',    train_loss.avg,               epoch)
			hparams.logger.add_scalar('Train/angle loss',   train_angle_loss.avg,         epoch)
			hparams.logger.add_scalar('Train/distance loss',train_distance_loss.avg,      epoch)
			hparams.logger.add_scalar('Val/pred loss',      val_loss.avg,               epoch)
			hparams.logger.add_scalar('Val/angle loss',     val_angle_loss.avg,         epoch)
			hparams.logger.add_scalar('Val/distance loss',  val_distance_loss.avg,      epoch)

			clean_hparam_directory(hparams)
			hparams_dict = {key: value for key, value in vars(hparams).items() if type(value) in [str, bool, int, float]}

			hparams.logger.add_hparams(hparams_dict, {'hparam/acc':val_loss.avg})


		'''Plotting the data at the end of each epoch'''

		if hparams.train_data in ['benzene_dft.npz',
					  'ethanol_dft.npz',
					  'malonaldehyde_dft.npz',
					  'H2O/HigherEnergy/H2O_HigherEnergy1.npz',
					  'H2O/HigherEnergy/H2O_HigherEnergy2.npz',
					  'H2O/HigherEnergy/H2O_HigherEnergy3.npz',
					  'H2O/LowerEnergy/H2O_LowerEnergy1.npz',
					  'Ethanol']:
			fig = plot_MD_data(hparams, batch_pred, batch_y, batch_val_pred, batch_val_y)
		elif hparams.train_data in ['hmc']:
			fig = plot_HMC_data(hparams, batch_pred, batch_y, batch_val_pred, batch_val_y)
		elif hparams.train_data in ['lorenz']:
			fig = plot_Lorenz_data(hparams, batch_pred, batch_y, batch_val_pred, batch_val_y)

		if hparams.log and hparams.plot: hparams.logger.add_image('predictions', matplotlibfigure_to_tensor(fig), epoch)

		if epoch%1==0 and hparams.plot: plt.show()

		plt.close()

	print(f'{hparams.model} {hparams.train_data} ' + desc)
	if hparams.log:
		logging.info(f'{hparams.model} {hparams.train_data} ' + desc)
