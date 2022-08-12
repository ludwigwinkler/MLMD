import os, sys
import argparse
import numpy as np
import matplotlib, warnings, numbers
import torch

file_path = os.path.dirname(os.path.abspath(__file__)) + '/MD_HyperparameterParser.py'
cwd = os.path.dirname(os.path.abspath(__file__)) # current directory PhD/MLMD/src

sys.path.append("/".join(cwd.split("/")[:-2])) # @PhD: detect MLMD folder

matplotlib.rcParams["figure.figsize"] = [10, 10]

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)


def str2bool(v):
	if isinstance(v, bool):
		return v
	elif type(v) == str:
		if v.lower() in ('yes', 'true', 't', 'y', '1'):
			return True
		elif v.lower() in ('no', 'false', 'f', 'n', '0'):
			return False
	elif isinstance(v, numbers.Number):
		assert v in [0, 1]
		if v == 1:
			return True
		if v == 0:
			return False
	else:
		raise argparse.ArgumentTypeError(f'Invalid Value: {type(v)}')

dataset_nicestr_dict = {'ethanol_dft.npz': 'Ethanol',
			'benzene_dft.npz': 'Benzene',
			'malonaldehyde_dft.npz': 'Malondialdehyde',
			'uracil_dft.npz': 'Uracil',
			'toluene_dft.npz': 'Toluene',
			'salicylic_dft.npz': 'Salicylic Acid',
			'naphthalene_dft.npz': 'Naphthalene',
			'paracetamol_dft.npz': 'Paracetamol',
			'aspirin_dft.npz': 'Aspirin',
			'keto_100K_0.2fs.npz': 'Keto-Malondialdehyde (100K, 0.2fs)',
			'keto_300K_0.2fs.npz': 'Keto-Malondialdehyde (300K, 0.2fs)',
			'keto_500K_0.2fs.npz': 'Keto-Malondialdehyde (500K, 0.2fs)',
			'cosine': 'Cosine DiffEq',
			'doublecosine': 'DoubleCosine DiffEq',
			'p_matrix': 'P Matrix',
			'ps_matrix': 'PS Matrix'
			}

model_nicestr_dict = {'lstm': 'LSTM',
		      'bi_lstm': 'Bi-LSTM',
		      'ode': 'NeuralODE',
		      'bi_ode': 'Bi-NeuralODE'}

def Interpolation_HParamParser(	logger=False,
				project='arandomproject',
				experiment=None,
				save_weights=True,
				fast_dev_run=False,
				dataset=[	'ethanol_dft.npz', 'benzene_dft.npz', 'malonaldehyde_dft.npz', 'uracil_dft.npz',
						'lorenz', 'hmc',
						'keto_100K_0.2fs.npz',
						'keto_300K_1.0fs.npz', 'keto_300K_0.2fs.npz',
						'keto_500K_0.2fs.npz'],
				model='lstm',
				diffeq_output_scaling=True,
				auto_scale_batch_size=True,
				batch_size=500,
				train_traj_repetition=1,
				plot=True, show=False,
				pct_dataset=1.0,
				num_layers=5,
				lr=-1,
				num_hidden_multiplier=10,
				input_length=3,
				output_length=5,
				output_length_train=-1,
				output_length_val=-1,
				output_length_sampling=False,
				criterion=['MSE', 'MAE'][1],
				interpolation=True,
				interpolation_mode='adiabatic',
				integration_mode='integrator',
				load_weights=False,
				limit_train_batches=2000,
				limit_val_batches=1000,
				max_epochs=2000,
				optim='adam'
			 ):

	if ('ode' in model or 'hamiltonian' in model) and input_length != 1:
		input_length = 1
		assert input_length == 1, f'ODE and HNN models can only work with input_length=1, but received {input_length}'

	assert 0 < pct_dataset <= 1.0, f'pct_dataset has to be [0,1], but is {pct_dataset}'

	hparams = argparse.ArgumentParser(description='parser example')
	# hparams = ModArgumentParser(description='parser example')

	hparams.add_argument('-logger', type=str2bool, default=logger)
	hparams.add_argument('-project', type=str, default=project)
	hparams.add_argument('-experiment', type=str, default=experiment)

	hparams.add_argument('-plot', type=str2bool, default=plot)
	hparams.add_argument('-show', type=str2bool, default=show)
	hparams.add_argument('-save_weights', type=str2bool, default=save_weights)
	hparams.add_argument('-load_weights', type=str2bool, default=load_weights)
	hparams.add_argument('-fast_dev_run', type=str2bool, default=fast_dev_run)

	hparams.add_argument('-num_workers', type=int, default=2 if torch.cuda.is_available() else 0)

	hparams.add_argument('-odeint', type=str, choices=['explicit_adams', 'fixed_adams' 'adams', 'tsit5', 'dopri5', 'euler', 'midpoint', 'rk4'],
			     default='rk4')
	hparams.add_argument('-model', 	type=str, choices=['hnn', 'ode', 'rnn', 'lstm', 'ode2', 'var',
							  'bi_ode', 'bi_hnn', 'bi_rnn','bi_lstm'],
			     		default=model)

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
								'keto_500K_0.2fs.npz',
								'p_matrix',
								'ps_matrix'],
			     default=dataset)
	hparams.add_argument('-pct_dataset', type=float, default=pct_dataset)

	hparams.add_argument('-num_hidden_multiplier', type=int, default=num_hidden_multiplier)
	hparams.add_argument('-num_layers', type=int, default=num_layers)
	hparams.add_argument('-interpolation', type=str2bool, default=interpolation)
	hparams.add_argument('-interpolation_mode', type=str, choices=['adiabatic', 'linear', 'transformer'], default=interpolation_mode)
	hparams.add_argument('-integration_mode', type=str, choices=['int', 'diffeq'], default=integration_mode)
	hparams.add_argument('-diffeq_output_scaling', type=str2bool, default=diffeq_output_scaling)

	hparams.add_argument('-max_epochs', type=int, default=max_epochs)
	hparams.add_argument('-limit_train_batches', type=int, default=limit_train_batches)
	hparams.add_argument('-limit_val_batches', type=int, default=limit_train_batches)
	hparams.add_argument('-optim', type=str, choices=['adam', 'sgd'], default=optim)
	hparams.add_argument('-batch_size', type=int, default=batch_size)
	hparams.add_argument('-auto_scale_batch_size', type=str2bool, default=auto_scale_batch_size)
	hparams.add_argument('-train_traj_repetition', type=int, default=train_traj_repetition)
	hparams.add_argument('-input_length', type=int, default=input_length)

	hparams.add_argument('-output_length', 		type=int, default=output_length)
	hparams.add_argument('-output_length_train', 	type=int, default=output_length_train)
	hparams.add_argument('-output_length_val', 	type=int, default=output_length_val)
	hparams.add_argument('-output_length_sampling', type=str2bool, default=output_length_sampling)

	hparams.add_argument('-criterion', type=str, default=criterion)
	hparams.add_argument('-lr', type=float, default=lr)

	hparams.add_argument('-val_split', type=float, default=0.9)  # first part is train, second is val batch_i.e. val_split=0.8 -> 80% train, 20% val

	hparams = hparams.parse_args()

	hparams.output_length_train = hparams.output_length if hparams.output_length_train == -1 else hparams.output_length_train
	hparams.output_length_val = hparams.output_length if hparams.output_length_val == -1 else hparams.output_length_val

	assert hparams.output_length_val > 1 and hparams.output_length_train > 1
	assert type(hparams.interpolation)==bool, f'{hparams.interpolation=}'
	assert hparams.interpolation_mode in ['adiabatic', 'linear', 'transformer'], f'{hparams.interpolation=}'

	if 'ode' in hparams.model or 'hnn' in hparams.model:
		if hparams.input_length !=1:
			print(f"Input length for {hparams.model} was {hparams.input_length}, changed to 1")
			hparams.input_length =1

	if hparams.experiment is None:
		experiment_str = f"{str(hparams.model)}_pct{str(hparams.pct_dataset)}_{str(hparams.dataset)}_Ttrain{str(hparams.output_length_train)}_Tval{str(hparams.output_length_val)}"
	else:
		experiment_str = f"{hparams.experiment}_{str(hparams.model)}_pct{str(hparams.pct_dataset)}_{str(hparams.dataset)}_Ttrain{str(hparams.output_length_train)}_Tval{str(hparams.output_length_val)}"
	hparams.__dict__.update({'experiment': experiment_str})
	hparams.__dict__.update({'ckptname': str(hparams.model)+'_'+str(hparams.dataset)+'_T'+str(hparams.output_length_train)})
	hparams.__dict__.update({'dataset_nicestr': dataset_nicestr_dict[hparams.dataset]})

	if hparams.dataset in ['p_matrix', 'ps_matrix']:
		hparams.__dict__.update({'system': 'statevec'})
		assert hparams.integration_mode == 'diffeq', f'Integrator integration not possible for statevec data'
	elif '_dft.npz' in hparams.dataset or 'keto' in hparams.dataset:
		hparams.__dict__.update({'system': 'dynsys'})
	else:
		raise AssertionError(f'System type (dynamical system or state vector) not defined for {hparams.dataset}')


	assert hparams.output_length >= 1
	assert hparams.output_length_train >= 1
	assert hparams.output_length_val >= 1

	return hparams

def Forecasting_HParamParser(	logger=False,
				project='arandomproject',
				experiment=None,
				save_weights=True,
				fast_dev_run=False,
				dataset=['cosine'],
				init_T = 1000,
				cosine_freq=1,
				T=2000,
				model='lstm',
				interpolation='adiabatic',
				integration_mode='integrator',
				subsampling=-1,
				batch_size=500,
				train_traj_repetition=1,
				plot=True, show=False,
				pct_dataset=1.0,
				num_layers=5,
				num_hidden_multiplier=10,
				input_length=3,
				output_length=5,
				output_length_train=-1,
				output_length_val=-1,
				val_split=0.8,
				output_length_sampling=False,
				criterion=['MSE', 'MAE'][1],
				load_weights=False,
				max_epochs=200,
				limit_train_batches=2000,
				optim='adam',
				dt=0.01
			 ):

	if ('ode' in model or 'hamiltonian' in model) and input_length != 1:
		input_length = 1
		assert input_length == 1, f'ODE and HNN models can only work with input_length=1, but received {input_length}'

	assert 0 < pct_dataset <= 1.0, f'pct_dataset has to be [0,1], but is {pct_dataset}'

	hparams = argparse.ArgumentParser(description='parser example')
	# hparams = ModArgumentParser(description='parser example')

	hparams.add_argument('-logger', type=str2bool, default=logger)
	hparams.add_argument('-project', type=str, default=project)
	hparams.add_argument('-experiment', type=str, default=experiment)

	hparams.add_argument('-plot', type=str2bool, default=plot)
	hparams.add_argument('-show', type=str2bool, default=show)
	hparams.add_argument('-save_weights', type=str2bool, default=save_weights)
	hparams.add_argument('-load_weights', type=str2bool, default=load_weights)
	hparams.add_argument('-fast_dev_run', type=str2bool, default=fast_dev_run)


	hparams.add_argument('-num_workers', type=int, default=4 if torch.cuda.is_available() else 0)

	hparams.add_argument('-odeint', type=str, choices=['explicit_adams', 'fixed_adams' 'adams', 'tsit5', 'dopri5', 'euler', 'midpoint', 'rk4'],
			     default='rk4')
	hparams.add_argument('-model', type=str, choices=['hnn', 'ode', 'rnn', 'lstm', 'ode2',
							  'bi_ode', 'bi_hnn', 'bi_rnn',
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
	hparams.add_argument('-pct_dataset', type=float, default=pct_dataset)
	hparams.add_argument('-subsampling', type=int, default=subsampling)
	hparams.add_argument('-init_T', type=int, default=init_T)
	hparams.add_argument('-dt', type=float, default=dt)
	hparams.add_argument('-cosine_freq', type=float, default=cosine_freq)
	hparams.add_argument('-T', type=int, default=T)

	hparams.add_argument('-num_hidden_multiplier', type=int, default=num_hidden_multiplier)
	hparams.add_argument('-num_layers', type=int, default=num_layers)
	hparams.add_argument('-interpolation', type=str, choices=['none', 'adiabatic', 'transformer'], default=interpolation)
	hparams.add_argument('-integration_mode', type=str, choices=['int', 'diffeq'], default=integration_mode)

	hparams.add_argument('-max_epochs', type=int, default=max_epochs)
	hparams.add_argument('-limit_train_batches', type=int, default=limit_train_batches)
	hparams.add_argument('-optim', type=str, choices=['adam', 'sgd'], default=optim)
	hparams.add_argument('-batch_size', type=int, default=batch_size)
	hparams.add_argument('-train_traj_repetition', type=int, default=train_traj_repetition)
	hparams.add_argument('-input_length', type=int, default=input_length)

	hparams.add_argument('-output_length', 		type=int, default=output_length)
	hparams.add_argument('-output_length_train', 	type=int, default=output_length_train)
	hparams.add_argument('-output_length_val', 	type=int, default=output_length_val)
	hparams.add_argument('-output_length_sampling', type=str2bool, default=output_length_sampling)

	hparams.add_argument('-criterion', type=str, default=criterion)

	hparams.add_argument('-val_split', type=float, default=val_split)  # first part is train, second is val batch_i.e. val_split=0.8 -> 80% train, 20% val

	hparams = hparams.parse_args()

	hparams.output_length_train = hparams.output_length if hparams.output_length_train == -1 else hparams.output_length_train
	hparams.output_length_val = hparams.output_length if hparams.output_length_val == -1 else hparams.output_length_val

	if 'ode' in hparams.model or 'hnn' in hparams.model:
		if hparams.input_length !=1:
			print(f"Input length for {hparams.model} was {hparams.input_length}, changed to 1")
			hparams.input_length =1

	if hparams.experiment is None:
		experiment_str = f"{str(hparams.model)}_{str(hparams.dataset)}_Ttrain{str(hparams.output_length_train)}_Tval{str(hparams.output_length_val)}"
	else:
		experiment_str = f"{hparams.experiment}_{str(hparams.model)}_{str(hparams.dataset)}_Ttrain{str(hparams.output_length_train)}_Tval{str(hparams.output_length_val)}"
	hparams.__dict__.update({'experiment': experiment_str})
	hparams.__dict__.update({'ckptname': str(hparams.model)+'_'+str(hparams.dataset)+'_TrainT'+str(hparams.output_length_train)})
	hparams.__dict__.update({'dataset_nicestr': dataset_nicestr_dict[hparams.dataset]})
	hparams.__dict__.update({'model_nicestr': model_nicestr_dict[hparams.model]})
	if hparams.dataset in ['p_matrix', 'ps_matrix']:
		hparams.__dict__.update({'system': 'statevec'})
	elif '_dft.npz' in hparams.dataset or 'cosine' in hparams.dataset:
		hparams.__dict__.update({'system': 'dynsys'})
	else:
		raise AssertionError(f'System type (dynamical system or state vector) not defined for {hparams.dataset}')

	assert hparams.output_length >= 1
	assert hparams.output_length_train >= 1
	assert hparams.output_length_val >= 1

	return hparams
