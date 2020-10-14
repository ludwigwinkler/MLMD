import future, sys, os, datetime, argparse, warnings, inspect
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

import scipy
import scipy as sp
from scipy.io import loadmat as sp_loadmat
import copy

cwd = os.path.abspath(os.getcwd())
os.chdir(cwd)

dir_path = 'dataefficiency'

files_in_dir = []

# r=>root, d=>directories, f=>files
for r, d, f in os.walk(dir_path):
	for item in f:

		files_in_dir.append(os.path.join(r, item))

print(f"{files_in_dir=}")

# exit()
for file_path in files_in_dir:
	if '/checkpoints/' in file_path and '.ckpt' in file_path:
		os.remove(file_path)