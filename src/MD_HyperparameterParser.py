import os, sys
import argparse
import numpy as np
import matplotlib, warnings
import torch

sys.path.append("/".join(os.getcwd().split("/")[:-1])) # experiments -> MLMD
from MLMD.src.MD_Utils import str2bool

matplotlib.rcParams["figure.figsize"] = [10, 10]

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


def HParamParser(	logger=False,
			logname='experiment',
			logdir='experimentdir',
			dataset=[	'ethanol_dft.npz', 'benzene_dft.npz', 'malonaldehyde_dft.npz', 'uracil_dft.npz',
					'lorenz', 'hmc',
					'keto_100K_0.2fs.npz',
				    	'keto_300K_1.0fs.npz', 'keto_300K_0.2fs.npz',
					'keto_500K_0.2fs.npz'],
			model='lstm',
			subsampling=-1,
			batchsize=10,
			train_traj_repetition=1,
			plot=True, show=False,
			pct_data_set=1.0,
			num_layers=5,
			input_length=3,
			output_length=5,
			output_length_train=-1,
			output_length_val=-1,
			output_length_sampling=False,
			criterion=['T', 't'][0],
			load_pretrained=False
		 ):
	if ('ode' in model or 'hamiltonian' in model) and input_length != 1:
		input_length = 1
		assert input_length == 1, f'ODE and HNN models can only work with input_length=1, but received {input_length}'

	assert 0 < pct_data_set <= 1.0, f'pct_data_set has to be [0,1], but is {pct_data_set}'

	hparams = argparse.ArgumentParser(description='parser example')
	# hparams = ModArgumentParser(description='parser example')

	hparams.add_argument('-logger', type=str2bool, default=logger)
	hparams.add_argument('-logname', type=str, default=logname)
	hparams.add_argument('-logdir', type=str, default=logdir)
	hparams.add_argument('-plot', type=str2bool, default=plot)
	hparams.add_argument('-show', type=str2bool, default=show)
	hparams.add_argument('-load_pretrained', type=str2bool, default=load_pretrained)

	hparams.add_argument('-num_workers', type=int, default=4 if torch.cuda.is_available() else 0)

	hparams.add_argument('-odeint', type=str, choices=['explicit_adams', 'fixed_adams' 'adams', 'tsit5', 'dopri5', 'euler', 'midpoint', 'rk4'],
			     default='rk4')
	hparams.add_argument('-model', type=str, choices=['hamiltonian', 'ode', 'rnn', 'lstm', 'bi_ode', 'bi_hamiltonian', 'bi_rnn',
							  'bi_lstm'], default=model)

	hparams.add_argument('-dataset', type=str, choices=[	'benzene_dft.npz',
							       	'ethanol_dft.npz',
							       	'malonaldehyde_dft.npz',
							       	'uracil_dft.npz',
							       	'toluene_dft.npz',
							       	'naphthalene_dft.npz',
							       	'salicylic_dft.npz',
							       	'paracetamol_dft.npz',
							       	'aspirin_dft.npz',
							       	'H2O/HigherEnergy/H2O_HigherEnergy1.npz',
							       	'H2O/HigherEnergy/H2O_HigherEnergy2.npz',
							       	'H2O/HigherEnergy/H2O_HigherEnergy3.npz',
							       	'H2O/LowerEnergy/H2O_LowerEnergy1.npz',
							       	'hmc',
							       	'lorenz',
							       	'keto_100K_0.2fs.npz',
							       	'keto_300K_0.2fs.npz','keto_300K_1.0fs.npz',
								'keto_500K_0.2fs.npz'],
			     default=dataset)
	hparams.add_argument('-pct_data_set', type=float, default=pct_data_set)
	hparams.add_argument('-subsampling', type=int, default=subsampling)

	hparams.add_argument('-num_hidden', type=int, default=-1)
	hparams.add_argument('-num_layers', type=int, default=num_layers)

	hparams.add_argument('-batchsize', type=int, default=batchsize)
	hparams.add_argument('-train_traj_repetition', type=int, default=train_traj_repetition)
	hparams.add_argument('-input_length', type=int, default=input_length)

	hparams.add_argument('-output_length', 		type=int, default=output_length)
	hparams.add_argument('-output_length_train', 	type=int, default=output_length_train)
	hparams.add_argument('-output_length_val', 	type=int, default=output_length_val)
	hparams.add_argument('-output_length_sampling', type=str2bool, default=output_length_sampling)

	hparams.add_argument('-criterion', type=str, default=criterion)

	hparams.add_argument('-val_split', type=float,
			     default=0.8)  # first part is train, second is val batch_i.e. val_split=0.8 -> 80% train, 20% val

	hparams = hparams.parse_args()

	hparams.output_length_train = hparams.output_length if hparams.output_length_train == -1 else hparams.output_length_train
	hparams.output_length_val = hparams.output_length if hparams.output_length_val == -1 else hparams.output_length_val

	if 'ode' in hparams.model or 'hamiltonian' in hparams.model:
		if hparams.input_length !=1:
			print(f"Input length for {model} was {input_length}, changed to 1")
			hparams.input_length =1

	hparams.__dict__.update({'logname': 	hparams.logname + '_' + str(hparams.model)
					    	+ '_pct' + str(hparams.pct_data_set) + '_' + str(hparams.dataset) +
			    			'_Ttrain' + str(hparams.output_length_train) + '_Tval' + str(hparams.output_length_val)})
	hparams.__dict__.update({'ckptname': 	str(hparams.model)+'_'+str(hparams.dataset)+'_TrainT'+str(output_length_train)})

	assert hparams.output_length >= 1
	assert hparams.output_length_train >= 1
	assert hparams.output_length_val >= 1

	return hparams
