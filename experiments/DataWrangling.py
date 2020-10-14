import future, sys, os, datetime, argparse
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

def array_to_Atoms(data):

	data = data[:,:data.shape[1]//2]
	# print(f"{data.shape=}")
	data = data.reshape(data.shape[0], -1, 3)

	# plt.plot(data[:200,0,:])
	# plt.show()
	frames = []

	for pos in data:
		frames += [Atoms('C6H6',positions=pos)]

	return frames



# data_npz = np.load('../data/aspirin_ccsd-train.npz')['R']
# print(data_npz.shape)
# frames = npz_to_xyz(data_npz)
# data_xyz = ase.io.xyz.read_xyz_trajectory()

# ase.io.xyz.write_xyz(fileobj='../data/yo.xyz', images=data_xyz)
# ase.io.xyz.write_xyz(fileobj='../data/yo.xyz', images=frames)

pred = torch.load('AnimationPred.pt')
y = torch.load('AnimationTrue.pt')
y0 = torch.load('AnimationConditions.pt')
y0 = y0[1:-1:3]
y0 = np.repeat(y0, y.shape[0] / y0.shape[0], axis=0)
# print(f"{y0.shape=}")
# print(f"{y.shape[0]/y0.shape[0]=}")
# print(y0[:100,0])
# print(y0[:100,0]-y[:100,0])
# t = torch.arange(0,100)[1:-1:3]


# print(t)
# exit()


ase.io.xyz.write_xyz(fileobj='BenzenePred.xyz', images=array_to_Atoms(pred.numpy()))
ase.io.xyz.write_xyz(fileobj='BenzeneTrue.xyz', images=array_to_Atoms(y.numpy()))
ase.io.xyz.write_xyz(fileobj='BenzeneConditions.xyz', images=array_to_Atoms(y0.numpy()))

