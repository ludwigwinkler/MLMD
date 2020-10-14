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

import wandb
api = wandb.Api()

# Change oreilly-class/cifar to <entity/project-name>
runs = api.runs("mlmd/BiDirectional_MD")
summary_list = []
config_list = []
name_list = []
for run in runs:
	# run.summary are the output key/values like accuracy.  We call ._json_dict to omit large files
	summary_list.append(run.summary._json_dict)

	# run.config is the input metrics.  We remove special values that start with _.
	config_list.append({k: v for k, v in run.config.items() if not k.startswith('_')})

	# run.name is the name of the run.
	name_list.append(run.name)

import pandas as pd

summary_df = pd.DataFrame.from_records(summary_list)
config_df = pd.DataFrame.from_records(config_list)
name_df = pd.DataFrame({'name': name_list})
all_df = pd.concat([name_df, config_df, summary_df], axis=1)

print(all_df)