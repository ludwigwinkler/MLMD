import future, sys, os, datetime, argparse
# print(os.path.dirname(sys.executable))
import torch
import numpy as np
import matplotlib
import random
from tqdm import tqdm
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

matplotlib.rcParams["figure.figsize"] = [10, 10]

import torch
import torch.nn.functional as F

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

import scipy
import scipy as sp
from scipy.io import loadmat as sp_loadmat
import copy

data = Tensor([0,1,2,3,4]).reshape(1,-1)
print(f'{data.shape=}')

data = data.repeat(7,1)

print(f'{data.shape=}')

data = torch.index_select(data, dim=-1, index=Tensor([2,1,0,4,3]).long())

print(data)

data = [0,1,2,3,4]
print(f'{data=}')
random.shuffle(data)
print(f'{data=}')



import random

list1_names = ['Jon', 'Emma', 'Kelly', 'Jason']
list2_salary = [7000, 6500, 9000, 10000]

print("Lists before Shuffling")
print("Employee Names: ", list1_names)
print("Employee Salaries: ", list2_salary)

# To Shuffle two List at once with the same order
mapIndexPosition = list(zip(list1_names, list2_salary))
random.shuffle(mapIndexPosition)
list1_names, list2_salary = zip(*mapIndexPosition)

print("Lists after Shuffling")
print("Employee Names: ", list1_names)
print("Employee Salary: ", list2_salary)

print("Employee name and salary present index 3")
print(list1_names[3], list2_salary[3])