import future, sys, os, datetime, argparse
# print(os.path.dirname(sys.executable))

import math
import numpy as np
import matplotlib
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


class Atom(torch.nn.Module):

	def __init__(self, data, hparams):

		self.atom_type = hparams.data_set

		super().__init__()
		if data.dim()==3: # data.shape=[batch, timesteps, atoms x (3x pos + 3x vel)
			'''data.shape=[batch, t, [atoms*pos, atoms*vel]] -> first split then reshape'''
			# if hparams.train_data_info=='pos':
			# 	self.num_atoms = data.shape[-1]//3
			# 	self.pos = data.reshape(data.shape[0], data.shape[1], data.shape[2] // 3, 3)  # shape = [BS, t, Atoms, cartesian dims]
			# elif hparams.train_data_info=='posvel':
			self.num_atoms = data.shape[-1]//6
			pos, vel = torch.chunk(data, chunks=2, dim=-1) # shape = [BS, t, Atoms*cartesian dims] for both pos & vel
			self.vel = vel.reshape(vel.shape[0], vel.shape[1], vel.shape[2]//3, 3) # shape = [BS, t, Atoms, cartesian dims]
			self.pos = pos.reshape(pos.shape[0], pos.shape[1], pos.shape[2]//3, 3) # shape = [BS, t, Atoms, cartesian dims]


		assert self.pos.dim()==4, f'self.pos.dim() should be [batch, timesteps, atom, 3 x pos ] but is {data.shape=}'

	@torch.no_grad()
	def compute_angle(self):

		if 'H2O' in self.atom_type:

			u = -self.pos[:,:,0,:] + self.pos[:,:,1,:]
			v = -self.pos[:,:,2,:] + self.pos[:,:,1,:]
			# torch.norm()
			linear_term = torch.sum(u*v,-1)/(u.norm(dim=-1)*v.norm(dim=-1))
			self.angle = torch.acos(linear_term.clamp(-1,1))
			self.degree = self.angle *360/(2*math.pi)
			return self.degree

		if 'benzene' in self.atom_type or 'malonaldehyde' in self.atom_type:

			u = -self.pos[:,:,0,:] + self.pos[:,:,1,:]
			v = -self.pos[:,:,2,:] + self.pos[:,:,1,:]
			# torch.norm()
			linear_term = torch.sum(u*v,-1)/(u.norm(dim=-1)*v.norm(dim=-1))
			self.angle = torch.acos(linear_term.clamp(-1,1))
			self.degree = self.angle *360/(2*math.pi)
			return self.degree

		# if self.atom_type in ['ethanol', 'Ethanol', 'ethanol_dft']:
		if 'ethanol' in self.atom_type.lower():
			'''
			def dih_angle(v1, v2, v3, v4):

				b1 = v2 - v1
				b2 = v3 - v2
				b3 = v4 - v3
			
				n1 = np.cross(b1, b2)
				n1 /= np.linalg.norm(n1)
			
				n2 = np.cross(b2, b3)
				n2 /= np.linalg.norm(n2)
			
				m1 = np.cross(n1, b2 / np.linalg.norm(b2))
			
				x = n1.dot(n2)
				y = m1.dot(n2)
			
				return np.arctan2(y, x)
				
			O-H rotor: 1-0-2-8
			H3 rotor: 7-1-0-2
			'''

			rotor = ['OH', 'H3'][1]
			if rotor == 'OH':
				v1 = self.pos[:,:,1]
				v2 = self.pos[:,:,0]
				v3 = self.pos[:,:,2]
				v4 = self.pos[:,:,8]
			elif rotor =='H3':
				v1 = self.pos[:, :, 7]
				v2 = self.pos[:, :, 1]
				v3 = self.pos[:, :, 0]
				v4 = self.pos[:, :, 2]

			b1 = v2 - v1
			b2 = v3 - v2
			b3 = v4 - v3

			n1 = torch.cross(b1, b2, dim=-1)
			n1 /= torch.norm(n1, dim=-1, keepdim=True)

			n2 = torch.cross(b2, b3, dim=-1)
			n2 /= torch.norm(n2, dim=-1, keepdim=True)

			m1 = torch.cross(n1, b2 / torch.norm(b2, dim=-1, keepdim=True), dim=-1)

			x = torch.sum(n1*n2, dim=-1)
			# print(f'{m1.shape=} {n2.shape=}')
			y = torch.sum(m1*n2, dim=-1)

			degree = torch.atan2(x, y)
			angle = degree #* 360 / (2 * math.pi)

			return angle

	def compute_distance(self):

		dist = self.pos.unsqueeze(-3) - self.pos.unsqueeze(-2) # expand around atoms and leave final dim untouched
		dist = torch.sum(dist**2, dim=-1)**0.5

		assert dist.shape==(*self.pos.shape[:2], self.num_atoms, self.num_atoms)

		return dist

	def compute_MD_geometry(self):

		angle 	= self.compute_angle()
		dist 	= self.compute_distance()

		return angle, dist



if __name__=='__main__':

	data = Tensor([[1,1,1,0,0,0],[0,0,0,1,1,1],[1,1,-1,2,2,2]])
	data = Tensor([[1,0,0,0],[0,0,0,0],[0,1,0,0]])
	data = data.reshape(1,1,*data.shape)

	atom = Atom(data)

	print(atom.compute_angle())
	print(atom.pos.requires_grad)
	print(atom.pos)

	optim = torch.optim.Adam(atom.parameters(), lr=0.01, betas=(0.66, 0.66))

	for i in range(100):
		optim.zero_grad()

		# angle = torch.sum(torch.abs(atom.compute_angle()))
		angle = torch.sum((atom.compute_angle())**2)
		angle.backward()

		print(f'Epoch: {i}')
		print(f'Angle: {angle.detach().item()}')
		print(f'{atom.pos.detach()=}')
		print()
		if i == 92:
			print(f'{torch.sum((atom.compute_angle()))}')
			print(atom.pos.grad)
			print(torch.sum((atom.compute_angle())**2))
		if torch.isnan(angle):

			break

		optim.step()

