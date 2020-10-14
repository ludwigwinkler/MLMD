import argparse
import numpy as np
import matplotlib, warnings
import torch
from Utils.Utils import str2bool

matplotlib.rcParams["figure.figsize"] = [10, 10]

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


def HParamParser(	logger=False,
			logname='experiment',
			logdir='experimentdir',
			data_set=['ethanol_dft.npz', 'benzene_dft.npz', 'malonaldehyde_dft.npz', 'uracil_dft.npz',
				    'lorenz', 'hmc',
				    'keto_1fs.npz', 'keto_0.2fs.npz'][-1],
			model='lstm',
			subsampling=-1,
			train_traj_repetition=1,
			plot=False,
			pct_data_set=1.0,
			num_hidden=200,
			num_layers=5,
			input_length=3,
			output_length=5,
			verbose=True,
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
	hparams.add_argument('-verbose', type=str2bool, default=verbose)

	hparams.add_argument('-gpus', type=int, default=1 if torch.cuda.is_available() else 0)
	hparams.add_argument('-num_workers', type=int, default=4 if torch.cuda.is_available() else 0)

	hparams.add_argument('-odeint', type=str, choices=['explicit_adams', 'fixed_adams' 'adams', 'tsit5', 'dopri5', 'euler', 'midpoint', 'rk4'],
			     default='rk4')
	hparams.add_argument('-model', type=str, choices=['hamiltonian', 'ode', 'rnn', 'lstm', 'bi_ode', 'bi_hamiltonian', 'bi_rnn',
							  'bi_lstm'], default=model)

	hparams.add_argument('-data_set', type=str, choices=[	'benzene_dft.npz',
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
							       	'keto_1fs.npz',
							       	'keto_0.2fs.npz'],
			     default=data_set)
	hparams.add_argument('-pct_data_set', type=float, default=pct_data_set)
	hparams.add_argument('-subsampling', type=int, default=subsampling)
	# hparams.add_argument('-num_samples', type=int, default=num_train_samples,
	# 		     help="Entire dataset: -1, First k samples: k e.g. '-num_samples 1000' gives you the first 1000 samples of the train data set")

	hparams.add_argument('-num_hidden', type=int, default=num_hidden)
	hparams.add_argument('-num_layers', type=int, default=num_layers)

	hparams.add_argument('-batch_size', type=int, default=200)
	hparams.add_argument('-train_traj_repetition', type=int, default=train_traj_repetition)
	hparams.add_argument('-input_length', type=int, default=input_length)
	hparams.add_argument('-output_length', type=int, default=output_length)

	hparams.add_argument('-plots_per_training', type=int, default=20)
	hparams.add_argument('-val_split', type=float,
			     default=0.8)  # first part is train, second is val batch_i.e. val_split=0.8 -> 80% train, 20% val

	hparams.add_argument('-val_prediction_steps', type=int, default=50)
	hparams.add_argument('-val_converge_criterion', type=int, default=20)
	hparams.add_argument('-val_per_epoch', type=int, default=200)

	hparams = hparams.parse_args()

	if 'ode' in hparams.model or 'hamiltonian' in hparams.model:
		if hparams.input_length !=1:
			print(f"Input length for {model} was {input_length}, changed to 1")
			hparams.inpupt_length =1

	return hparams
