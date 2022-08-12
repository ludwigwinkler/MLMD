import urllib

import future, sys, os, datetime
# print(os.path.dirname(sys.executable))
import torch
import numpy as np
import matplotlib
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch
from torch.nn import Module, Parameter
from torch.nn import Linear, Tanh, ReLU
import torch.nn.functional as F

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

sys.path.append("../../..")  # Up to -> KFAC -> Optimization -> PHD

import ase, ase.io.xyz
from ase import Atoms

cwd = os.path.abspath(os.getcwd())
os.chdir(cwd)

atom_order = {'aspirin_dft.npz': 'CCCCCCCOOOCCOHHHHHHHH',
	      'benzene_dft.npz': 'CCCCCCHHHHHH',
	      'ethanol_dft.npz': 'CCOHHHHHH',
	      'malonaldehyde_dft.npz': 'CCCOOHHHH',
	      'naphthalene_dft.npz': 'CCCCCCCCCCHHHHHHHH',
	      'paracetamol_dft.npz': 'CCONCCCCOCCHHHHHHHHH',
	      'toluene_dft.npz': 'CCCCCCCHHHHHHHH',
	      'uracil_dft.npz': 'CCNCNCOOHHHH',
	      'salicylic_dft.npz': 'CCCOCCCCOOHHHHHH'}

# atom_order = {'aspirin': 'CCCCCCCOOOCCOHHHHHHHH',
# 	      'benzene': 'CCCCCCHHHHHH',
# 	      'ethanol': 'CCOHHHHHH',
# 	      'malonaldehyde': 'CCCOOHHHH',
# 	      'naphthalene': 'CCCCCCCCCCHHHHHHHH',
# 	      'paracetamol': 'CCONCCCCOCCHHHHHHHHH',
# 	      'toluene': 'CCCCCCCHHHHHHHH',
# 	      'uracil': 'CCNCNCOOHHHH'}

def download_xyz_files():
	for molecule_url in ['http://quantum-machine.org/gdml/data/xyz/benzene_old_dft.zip',
			 'http://quantum-machine.org/gdml/data/xyz/toluene_dft.zip',
			 'http://quantum-machine.org/gdml/data/xyz/malonaldehyde_dft.zip',
			 'http://quantum-machine.org/gdml/data/xyz/ethanol_dft.zip',
			 'http://quantum-machine.org/gdml/data/xyz/paracetamol_dft.zip',
			 'http://quantum-machine.org/gdml/data/xyz/aspirin_dft.zip',
			 'http://quantum-machine.org/gdml/data/xyz/uracil_dft.zip',
			 'http://quantum-machine.org/gdml/data/xyz/naphthalene_dft.zip']:


		print(f'Downloading {molecule_url} from quantum-machine.org/gdml/data/xyz')
		urllib.request.urlretrieve(molecule_url, molecule_url.split('/')[-1])

def read_xyz_trajectory(str='../data/aspirin_ccsd-train.xyz'):
	data = []
	frame = 0
	end_of_frames = False
	while not end_of_frames:
		try:
			data += [x for x in ase.io.xyz.read_xyz(str, index=frame)]
			frame += 1
		except:
			end_of_frames = True

	return data


def write_xyz(fileobj, images, comment='', fmt='%22.15f'):
	comment = comment.rstrip()
	if '\n' in comment:
		raise ValueError('Comment line should not have line breaks.')
	for atoms in images:
		natoms = len(atoms)
		fileobj.write('%d\n%s\n' % (natoms, comment))
		for s, (x, y, z), (vx, vy, vz) in zip(atoms.symbols, atoms.positions, atoms.arrays['momenta']):
			fileobj.write('%-2s %s %s %s %s %s %s \n' % (s, fmt % x, fmt % y, fmt % z,
								     	fmt % vx, fmt % vy, fmt % vz))


def array_to_Atoms(data, molecule=None):
	assert molecule is not None

	molecule = atom_order[molecule] # dataset string -> atom order
	pos, vel = np.split( data, indices_or_sections=2, axis=-1)
	# print(f"{data.shape=}")
	pos = pos.reshape(pos.shape[0], -1, 3)
	vel = vel.reshape(vel.shape[0], -1, 3) *100

	# plt.plot(data[:200,0,:])
	# plt.show()
	frames = []

	for pos_, vel_ in zip(pos, vel):
		frames += [Atoms(molecule,positions=pos_, momenta=vel_)]

	return frames

def export_to_xyz(y, pred, y0, t0, molecule=None, filename=None, path='MDPredictions'):
	assert filename is not None

	y = y[:20000]
	pred = pred[:20000]
	t0 = t0[t0 < 20000]
	y0 = y0[:t0.shape[0]]

	write_xyz(fileobj=open(f'{path}/{filename}Pred.xyz', 'w'), images=array_to_Atoms(pred.numpy(), molecule=molecule))
	write_xyz(fileobj=open(f'{path}/{filename}True.xyz', 'w'), images=array_to_Atoms(y.numpy(), molecule=molecule))
	write_xyz(fileobj=open(f'{path}/{filename}Conditions.xyz', 'w'), images=array_to_Atoms(y0.numpy(), molecule=molecule))


if __name__=='__main__':

	y, pred, y0, t0 = torch.load('SalicylicAcid_BiLSTM_T20.pt').values()

	y = y[:20000]
	pred = pred[:20000]
	t0 = t0[t0<20000]
	y0 = y0[:t0.shape[0]]


	write_xyz(fileobj=open('SalicylicAcidBiLSTMT20Pred.xyz', 'w'), images=array_to_Atoms(pred.numpy()))
	write_xyz(fileobj=open('SalicylicAcidBiLSTMT20True.xyz', 'w'), images=array_to_Atoms(y.numpy()))
	write_xyz(fileobj=open('SalicylicAcidBiLSTMT20Conditions.xyz', 'w'), images=array_to_Atoms(y0.numpy()))
	#

	# download_xyz_files()


