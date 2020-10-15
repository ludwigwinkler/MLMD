"""
This module contains all routines for the integrator.
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
from tqdm import tqdm
from Gupta_PyTorch import GuptaTorch
from Gupta_PyTorch import AtomType as GuptaParamsDict

#!##########################################################################
#! Parametros de ambiente
#eV_J= 1.602176487e-19 #  Joules = N*m = kg*m^2/s^2}
from ase.units import _e as eV_J
#uma = 1.660538782e-27 # kg = 931.494028d6    eV/c^2   NIST
from ase.units import _amu as uma
#k_B = 8.617343e-5     # Boltzmann constant in eV/K     NIST
from ase.units import kB as k_B
from ase.units import kcal, mol
fs = 1.e-15    # femto segundos
Ang = 1.e-10   # angstroms
kcalmol_2_J = kcal * eV_J  # Joules
energyMDunits  = eV_J*fs**2/(uma*Ang**2) # uma*Ang^2/fs^2
#!##########################################################################

class VerletVelocity_Torch(nn.Module):
    """
    PyTorch version of Gupta force field. Derives from
    :class:`torch.nn.Module`.
    """
    def __init__(self, AtmType=None, n_atms=1, dt=1.0, units='eV'):
        """
        Parameters
        ----------
        model : Dictionary
            Loaded from main.
        n_atms : int
            Loaded from main.
        """

        super(VerletVelocity_Torch, self).__init__()

        self.AtmType = AtmType
        self.n_atms = n_atms

        self.GuptaFuntions = GuptaTorch(self.AtmType, self.n_atms)
        self.PredictGupta_E_F = self.GuptaFuntions.PredictGupta_E_F

        self.mass = self.GuptaFuntions.mass
        self.dt = dt

        # Ctes used in Verlet
        self.cteCoordA = self.dt /self.GuptaFuntions.r0
        self.cteCoordB = 0.5 * energyMDunits * self.dt ** 2 / (self.mass * self.GuptaFuntions.r0)
        self.cteVerletVelA = 0.5 * self.dt * energyMDunits / self.mass
        self.cteVerletVelB = self.dt / self.GuptaFuntions.r0

    #! ---------Verlet velocity----------------
    def VerletVelocity_integrator(self, COORDold, VELold, FUERold):
        #! cteA=0.5*dt*eV/mass
        #! cteB=dt
        VEL_aux = VELold + self.cteVerletVelA * FUERold             #! Ang/fs
        COORD = COORDold + self.cteVerletVelB * VEL_aux             #! en terminos de r0
        ENER, FUER = self.GuptaFuntions.PredictGupta_E_F(COORD)
        VEL = VEL_aux + self.cteVerletVelA * FUER 
        return COORD, VEL, FUER, ENER
    #! ----------------------------------------

    #! ----------------------------------------
    def CoordsTaylorSerie(self, Xold, Vel, Fza):
        #! cteA=dt
        #! cteB=0.5d0*eV*dt*dt/(mass)
        return Xold + self.cteCoordA * Vel + self.cteCoordB * Fza
    #! ----------------------------------------

    #! ----------------------------------------
    def Main_evolution_function(self, X0, V0, AtmTyp, units):

        # Initialize MD: Xold = X(t-1), X = X(t), Xnew = X(t+1)
        E0, F0 = self.PredictGupta_E_F(X0)
        X = self.CoordsTaylorSerie(self, X0, V0, F0)
        #print(X.view(-1))
        #print(torch.dot(X.view(-1),X.view(-1)))

        #! *** Archivos de salida ***
        file_OUT_X = open("cordenadas.xyz",'w')
        file_OUT_V = open("velocidades.xyz",'w')
        #file_OUT_prom = open("promedios.dat",'w')
        #file_OUT_prom.write("#Tiempo(ps) Temp(K) EnergiaCin(eV) EnergiaPot(eV) EnergTot(eV) 1/EnergiaCin(1/eV) EnergTot^2(eV^2)\n")

        #! *** Ciclo main ***
        print("*** Ciclo main ***")
        contador = 0
        
        COORD = torch.zeros_like(X0)
        VEL = torch.zeros_like(V0)
        FUER = torch.zeros_like(F0)
        #ENERGIASPROMEDIO = torch.zeros(5)
        #EnergPromFinal = torch.zeros(5)

        for i in tqdm(range(int(Npas/N_esc))):
            #ENERGIASPROMEDIO.fill_(0.0)
            for j in range(N_esc):
                contador += 1
            
                COORD, VEL, FUER, Ecoh = self.VerletVelocity_integrator(X0, V0, F0)
                
                #ENERGIAS, ENERGIASPROMEDIO = _calculaEnergias(Ecoh, cteKin * torch.dot(VEL.view(-1),VEL.view(-1)),ENERGIASPROMEDIO)
                
                COORDold = COORD.clone().detach()
                VELold   = VEL.clone().detach()
                FUERold  = FUER.clone().detach()

            #EnergPromFinal += ENERGIASPROMEDIO
            #ENERGIASPROMEDIO /= N_esc
            
            if ENTRADA[0] == 1: 
                X_tmp = COORD.cpu().numpy()
                file_OUT_X.write('%d\nStep: %d\n'%(n_atms,contador))
                for atm_i in range(n_atms):
                    file_OUT_X.write('%s %10.6f %10.6f %10.6f\n'%(typ[atm_i],X_tmp[atm_i,0],X_tmp[atm_i,1],X_tmp[atm_i,2]))
            if ENTRADA[1] == 1: 
                X_tmp = VEL.cpu().numpy()
                file_OUT_V.write('%d\nStep: %d\n'%(n_atms,contador))
                for atm_i in range(n_atms):
                    file_OUT_V.write('%s %10.6f %10.6f %10.6f\n'%(typ[atm_i],X_tmp[atm_i,0],X_tmp[atm_i,1],X_tmp[atm_i,2]))
            
            #! Boltzmann constant in eV/K     NIST
            #EPROM_tmp = ENERGIASPROMEDIO.cpu().numpy()
            #T = 2 * EPROM_tmp[0] / ( k_B * (3 * n_atms - 6))
            #file_OUT_prom.write('%12.4f %10.3f %10.6f %10.6f %10.6f %10.6f %10.6f\n'%(contador*dt*1000,T,EPROM_tmp[0],EPROM_tmp[1],EPROM_tmp[2],EPROM_tmp[3],EPROM_tmp[4]))

        file_OUT_X.close()
        file_OUT_V.close()
        #file_OUT_prom.close()

        return "DONE!"
    #! ----------------------------------------

# -------- Read xyz files ------------
def readxyz(filename):
    atoms = []
    coordinates = []
    xyz = open(filename, "r")
    n_atoms = int(xyz.readline())
    title = xyz.readline()
    for _ in range(n_atoms):
        line = xyz.readline()
        atom, x, y, z = line.split()
        atoms.append(atom)
        coordinates.append(float(x))
        coordinates.append(float(y))
        coordinates.append(float(z))
    xyz.close()
    return n_atoms, atoms, torch.cuda.DoubleTensor(coordinates)
#! ----------------------------------------
