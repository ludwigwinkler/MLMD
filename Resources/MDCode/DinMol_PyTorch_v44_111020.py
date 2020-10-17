import numpy as np
import torch 
from tqdm import tqdm
import sys

sys.path.append("../../..")  # MDCode -> resources -> PhD

from MLMD.resources.MDCode.Gupta_PyTorch import GuptaTorch
from MLMD.resources.MDCode.Gupta_PyTorch import AtomType as GuptaParamsDict
from MLMD.resources.MDCode.intengrators import VerletVelocity_Torch
from MLMD.resources.MDCode.intengrators import readxyz



# ----------- Initialize PyTorch ----------------
dtype = torch.double
torch.set_default_dtype(dtype)
use_gpu = torch.cuda.is_available() #else "cpu"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
if use_gpu:
    torch.set_default_tensor_type(torch.cuda.DoubleTensor)
    print(torch.cuda.device_count())
    print(torch.cuda.is_available())
else:
    torch.set_default_tensor_type(torch.DoubleTensor)
# device = torch.device("cuda:0") # Uncomment torch.s to run on GPU
print('Using device:',device)
# ------------------------------------------------

##########################################################################
#! Main Program
##########################################################################

# Initial coordinates
n_atms, AtmTyp, X0 = readxyz("coord_ini.xyz")#Au55-ICO-ang.equil.ase.xyz")
# initial velocities
n_atms, AtmTyp, V0 = readxyz("vel_ini.xyz")
V0 = torch.zeros_like(X0)
print("Coords and velocities read.")
print("Number of atoms:", n_atms)

# *** Read input file ***
paramsMD = open("input_parameters.dat", 'r')
line = paramsMD.readline()
#dt(fs)  Npas   M_esc   beta    gamma0
dt, Npas, N_esc, beta, gamma0 = list(map(float, paramsMD.readline().split()))
_ = paramsMD.readline()
line = paramsMD.readline().split()
#coord   velocidad   distancias   promedios;  1=Si,0=No 
ENTRADA=[int(i) for i in line]
# Energy and force units
_ = paramsMD.readline()
line = paramsMD.readline().split()
units = [int(i) for i in line]


#! ------- Inicializa las variables ------
#! *** Re-escala las coordenadas ***
X0 *= gamma0
#! *** Vf=beta*Vi -> Tf=beta^2 *Ti ***
V0 *= beta #Ang/fs

X0 = X0.view(-1, 3)
V0 = V0.view(-1, 3)

VV_integrator = VerletVelocity_Torch(AtmTyp, n_atms, dt=dt, units='eV')

status = VV_integrator.Main_evolution_function(X0, V0, AtmTyp)

print(status)