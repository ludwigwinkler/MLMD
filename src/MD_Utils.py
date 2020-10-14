import numpy as np
import argparse, sys, os, shutil

import torch
import torch.nn.functional as F

def str2bool(v):
	if isinstance(v, bool):
		return v
	if v.lower() in ('yes', 'true', 't', 'y', '1'):
		return True
	elif v.lower() in ('no', 'false', 'f', 'n', '0'):
		return False
	else:
		raise argparse.ArgumentTypeError('Boolean value expected.')

def clean_hparam_directory(params):

	logpath = params.logpath+'/'
	hparam_files = [f for f in os.listdir(logpath) if os.path.isdir(os.path.join(logpath, f))]
	[shutil.rmtree(logpath+hparam_file, ) for hparam_file in hparam_files]

def matplotlibfigure_to_tensor(fig):

	fig.canvas.draw()
	array = np.frombuffer(fig.canvas.tostring_rgb(), dtype=np.uint8)
	array = array.reshape(fig.canvas.get_width_height()[::-1] + (3,))
	# tensor = torch.from_numpy(array).permute(2,0,1)
	tensor = array.transpose(2,0,1)

	return tensor

def NormalizedLoss(pred, target):

	assert pred.dim()==3
	assert pred.shape==target.shape

	target_mu = target.mean(dim=[0,1], keepdim=True)
	target_std = target.std(dim=[0,1], keepdim=True)

	target_	= (target - target_mu)/(target_std)
	pred_ 	= (pred - target_mu)/(target_std)

	return F.mse_loss(pred_, target_)
