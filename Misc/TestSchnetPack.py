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
import torch.nn.functional as F

import ase
from schnetpack import AtomsData

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

import scipy
import scipy as sp
from scipy.io import loadmat as sp_loadmat
import copy

atoms = ase.io.read('../data/ethanol.xyz', index=':100')

pos = []
for atom in atoms:
	pos.append(atom.get_positions())

pos = np.array(pos)
vel = pos[1:] - pos[:-1]

print(f'{pos.shape=}')
print(f'{vel.shape=}')

fig, axs = plt.subplots(1,2, sharex=True)
axs = axs.flatten()

axs[0].plot(pos[:,0,:])
axs[1].plot(vel[:,0,:])
plt.show()