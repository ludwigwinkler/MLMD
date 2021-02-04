import numpy as np
import argparse, sys, os, shutil, time

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


class Timer:
	"""Record multiple running times."""

	def __init__(self):
		self.times = []
		self.start()

	def start(self):
		self.tik = time.time()

	def stop(self):
		# Stop the timer and record the time in a list
		self.times.append(time.time() - self.tik)
		return self.times[-1]

	def avg(self):
		# Return the average time
		return sum(self.times) / len(self.times)

	def sum(self):
		# Return the sum of time
		return sum(self.times)

	def cumsum(self):
		# Return the accumulated times
		return np.array(self.times).cumsum().tolist()

class Benchmark:
	def __init__(self, description='Done in %.4f sec', repetitions=1):
		'''
		repetitions = 3
		with Benchmark():
			 for _ in range(3):
			 	run_some_code_xyz()
		-> prints descriptions with running_time/repetitions
		'''
		self.description = description
		self.repetitions = repetitions

	def __enter__(self):
		self.timer = Timer()
		return self

	def __exit__(self, *args):
		print(self.description % (self.timer.stop() / self.repetitions))
