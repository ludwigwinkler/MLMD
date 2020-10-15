"""
This module contains all routines for evaluating Gupta FFs.
"""

# MIT License
#
# Copyright (c) 2019 Huziel E. Sauceda
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import numpy as np
import torch
import torch.nn as nn

#! _______________________________________________________
#!|_____________________TipoAtomo_________________________|
#![epsilon0] = eV
#![zeta0] = eV
#![p] = no unidades
#![q] = no unidades
#![z] = numero atomico
#![mass] = uma
#![r0] = Angstroms, distancia primeros vecinos
AtomType={'Na':{'source': 'Bliesten', 'epsilon0': 0.015955395,'zeta0': 0.2911346,'p': 10.13,'q': 1.3,'z': 11,'mass': 22.98976928,'r0': 3.6989} \
        ,'Al':{'source': 'Cleri-Rosato 1993','epsilon0': 0.1221,'zeta0': 1.316,'p':8.612,'q': 2.516,'z': 13,'mass': 26.9815386,'r0': 2.863} \
        ,'Co':{'source': '','epsilon0': 0.0950,'zeta0': 1.4880,'p': 11.604,'q': 2.286,'z': 27,'mass': 58.933195,'r0': 1}\
        ,'Ni':{'source': '','epsilon0': 0.084994 ,'zeta0': 1, 'p': 10.0,'q': 2.7,'z': 28,'mass': 58.6934, 'r0': 2.491}\
        ,'Cu':{'source': 'Cleri-Rosato 1993','epsilon0': 0.0855,'zeta0': 1.224,'p': 10.960 , 'q': 2.278,'z': 29,'mass': 63.546 ,'r0': 2.556 }\
        ,'Zn':{'source': '','epsilon0': 0.1477,'zeta0': 0.8900, 'p': 9.689,'q': 4.602 ,'z': 30,'mass': 65.38 ,'r0': 2.665 }\
        ,'Rh':{'source': '','epsilon0': 0.0629 ,'zeta0': 1.66, 'p': 18.45,'q': 1.867 ,'z': 45,'mass': 102.90550 ,'r0': 1}\
        ,'Pd':{'source': 'Cleri-Rosato 1993','epsilon0': 0.1746, 'zeta0': 1.718,'p': 10.867 , 'q': 3.742, 'z'  : 46,'mass'  : 106.42 ,'r0': 2.751 }\
        ,'Ag':{'source': 'Cleri-Rosato 1993','epsilon0': 0.1028,'zeta0': 1.178 ,'p': 10.928 ,'q': 3.139,'z'  : 47, 'mass'  : 107.8682 , 'r0': 2.889 }\
        ,'Cd':{'source': '','epsilon0': 0.1420, 'zeta0': 0.8117,'p': 10.612, 'q': 5.206 ,'z': 48,'mass': 112.411,'r0': 1}\
        ,'Te':{'source': '','epsilon0': 0.1420, 'zeta0': 0.8117,'p': 10.4,'q': 5.206 , 'z': 52,'mass': 127.60 ,'r0': 1}\
        ,'Ir':{'source': 'Cleri-Rosato 1993','epsilon0': 0.1156,'zeta0': 2.289, 'p': 16.98,'q': 2.691,'z': 77, 'mass': 192.217 ,'r0': 2.715 }\
        ,'Pt':{'source': 'Cleri-Rosato 1993','epsilon0': 0.2975,'zeta0': 2.695,'p': 10.612, 'q': 4.004, 'z': 78,'mass': 195.084, 'r0': 2.775 }\
        ,'Au':{'source': 'Cleri-Rosato 1993','epsilon0': 0.2061,'zeta0': 1.790,'p': 10.229,'q': 4.036,'z': 79,'mass': 196.966569 ,'r0': 2.884 }\
        ,'Pb':{'source': 'Cleri-Rosato 1993','epsilon0': 0.0980,'zeta0': 0.914,'p': 9.576, 'q': 3.648, 'z': 82, 'mass': 207.2,'r0':3.501}}
#________________________________________________________|

class GuptaTorch(nn.Module):
    """
    PyTorch version of Gupta force field. Derives from
    :class:`torch.nn.Module`.
    """
    def __init__(self, model, n_atms):
        """
        Parameters
        ----------
        model : Dictionary
            Loaded from main.
        n_atms : int
            Loaded from main.
        """

        super(GuptaTorch, self).__init__()

        self.GuptaParams = AtomType[model]
        self.n_atms = n_atms

        self.mass = self.GuptaParams['mass']

        # Force costants
        self.cteFzA = 2*self.GuptaParams['epsilon0']*self.GuptaParams['p']*np.exp(self.GuptaParams['p'])
        self.cteFzB = self.GuptaParams['zeta0']*self.GuptaParams['q']*np.exp(self.GuptaParams['q'])
        self.cteFzC = -2*self.GuptaParams['q']
        self.cteFzD = -self.GuptaParams['p']
        # Energy ctes
        self.cteUcohA = self.GuptaParams['epsilon0']*np.exp(self.GuptaParams['p'])
        self.cteUcohB = self.GuptaParams['zeta0']*np.exp(self.GuptaParams['q'])
        self.r0 = self.GuptaParams['r0']

        self.I_nxn = torch.eye(self.n_atms)#, device=self.device)

    def PredictGupta_E_F(self, X):

        X = X / self.r0  # r_ij/r0 -> r'_ij
        r_ij = X[None, :] - X[:, None] # NxNx3
        dists = r_ij.norm(dim=-1)      # NxN

        M_exp_p  = torch.exp(self.cteFzD*dists) - self.I_nxn
        M_exp_2q = torch.exp(self.cteFzC*dists) - self.I_nxn

        Phi_p  = torch.sum(M_exp_p, dim=0) # size = [1, ncol]
        Phi_2q = torch.sum(M_exp_2q, dim=0)
        inv_sqrt_Phi_2q = torch.reciprocal(torch.sqrt(Phi_2q))
        M_inv_gammas = inv_sqrt_Phi_2q[None, :] + inv_sqrt_Phi_2q[:, None]

        # Computes energy
        E = torch.sum(self.cteUcohA * Phi_p - self.cteUcohB * torch.sqrt(Phi_2q))
        # Computes forces
        fij = torch.div(torch.addcmul( self.cteFzA * M_exp_p , -self.cteFzB , M_exp_2q , M_inv_gammas) , self.r0 * dists + self.I_nxn)
        fij = fij.expand(3, self.n_atms , self.n_atms).permute(1,2,0)

        return E, torch.sum(fij*r_ij, dim=0) # eV, eV/Ang