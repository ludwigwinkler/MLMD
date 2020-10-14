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
import pytorch_lightning as light
from torch.nn import Module, Parameter
from torch.nn import Linear, Tanh, ReLU
import torch.nn.functional as F

Tensor = torch.Tensor
FloatTensor = torch.FloatTensor

torch.set_printoptions(precision=4, sci_mode=False)
np.set_printoptions(precision=4, suppress=True)

sys.path.append("../../..")  # Up to -> KFAC -> Optimization -> PHD

from argparse import ArgumentParser

cwd = os.path.abspath(os.getcwd())
os.chdir(cwd)

import os

import torch
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision import transforms

import pytorch_lightning as light
from pytorch_lightning import LightningModule
from pytorch_lightning.loggers import TensorBoardLogger
from pytorch_lightning.callbacks import EarlyStopping
from pytorch_lightning.metrics.functional import accuracy
from pytorch_lightning import Trainer, seed_everything
# seed_everything(0)

from argparse import ArgumentParser

class LightningModel(LightningModule):

	def __init__(self, classes=10):
		super().__init__()
		self.save_hyperparameters()
		# print(f'{self.hparams=}')

		self.net = torch.nn.Sequential(Linear(28*28, self.hparams.classes.num_hidden_units), ReLU(),
					       Linear(self.hparams.classes.num_hidden_units,self.hparams.classes.num_hidden_units), ReLU(),
					       Linear(self.hparams.classes.num_hidden_units,self.hparams.classes.num_hidden_units), ReLU(),
					       Linear(self.hparams.classes.num_hidden_units, 10))

	@staticmethod
	def add_model_specific_args(parent_parser):
		'''
		Adds arguments to the already existing argument parser 'parent_parser'
		:param parent_parser:
		:return:
		'''
		parser = ArgumentParser(parents=[parent_parser], add_help=False)
		parser.add_argument('--num_layers', type=int, default=4)
		parser.add_argument('--num_hidden_units', type=int, default=300)
		return parser

	def forward(self, x):
		# return torch.relu(self.l1(x.view(x.size(0), -1)))
		return self.net(x.view(x.size(0), -1))

	def training_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		acc = accuracy(y_hat, y)
		loss = F.cross_entropy(y_hat, y)
		tqdm_dict = {'acc': float(acc)}
		output = {'loss': loss, 'progress_bar': tqdm_dict}
		return output

	def validation_step(self, batch, batch_idx):
		x, y = batch
		y_hat = self(x)
		loss = F.cross_entropy(y_hat, y)
		acc = accuracy(y_hat, y)
		return {'val_loss': loss, 'val_acc':acc}

	def validation_epoch_end(self, outputs):
		avg_loss = torch.stack([x['val_loss'] for x in outputs]).mean()
		avg_acc = torch.stack([x['val_acc'] for x in outputs]).mean()
		logs = {'val/loss': avg_loss, 'val/accuracy': avg_acc}
		return {'val_loss': avg_loss, 'progress_bar': {'val_accuracy': avg_acc}, 'log':logs}

	def configure_optimizers(self):
		return torch.optim.Adam(self.parameters(), lr=0.001)

	def on_fit_start(self):
		metric_placeholder = {'val/accuracy': 0}
		self.logger.log_hyperparams(self.hparams, metrics=metric_placeholder)

parser = ArgumentParser()
# add PROGRAM level args
parser.add_argument('--name', type=str, default='some_name')
parser = LightningModel.add_model_specific_args(parser) # static method
args = parser.parse_args()



# data
# mnist_train = MNIST('../../../Data/MNIST', train=True, download=True, transform=transforms.ToTensor())
# mnist_val = MNIST('../../../Data/MNIST', train=False, download=True, transform=transforms.ToTensor())
fmnist_train = MNIST('../../../Data/FMNIST', train=True, download=True, transform=transforms.ToTensor())
fmnist_val = MNIST('../../../Data/FMNIST', train=False, download=True, transform=transforms.ToTensor())
mnist_train = DataLoader(fmnist_train, batch_size=64)
mnist_val = DataLoader(fmnist_val, batch_size=64)

model = LightningModel(args)

logger = TensorBoardLogger("tb_logs", name="my_model")
early_stop_callback = EarlyStopping(
	monitor='val_accuracy',
	min_delta=0.00,
	patience=3,
	verbose=True,
	mode='max'
)

# most basic trainer, uses good defaults
trainer = Trainer(max_epochs=20, progress_bar_refresh_rate=100,
		  limit_train_batches=0.3,
		  early_stop_callback=early_stop_callback
		  )

trainer.fit(model, mnist_train, mnist_val)

