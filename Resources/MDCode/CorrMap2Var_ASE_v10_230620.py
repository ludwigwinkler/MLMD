import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from math import *
from pylab import *
from scipy.stats import norm
import matplotlib.mlab as mlab
import sys,os
import time

import h5py

from ase import Atoms
from ase.data import atomic_numbers

from tqdm import tqdm
 
# |********************************************|
# |***    Start functions definitions       ***|
# |********************************************|

#def AtomTyp(z): return list(atomic_numbers.keys())[list(atomic_numbers.values()).index(z)]

def Permutes_ij(Perm,X):#,F):
    x_new = X.copy()
    #F_new = np.zeros(shape=F.shape)
    # --- Invert indices: j -> i
    for atm_index in range(n_atoms):
        #print Perm[atm_index], "->", atm_index
        x_new[atm_index] = X[Perm[atm_index]]
        #F_new[atm_index] = F[Perm[atm_index]]
    return x_new#,F_new

def AtomTyp(z_typ): 
    if 'int' in str(type(z_typ)): return list(atomic_numbers.keys())[list(atomic_numbers.values()).index(z_typ)]
    elif 'numpy.bytes_' in str(type(z_typ)): return z_typ.decode('UTF-8')
    elif 'str' in str(type(z_typ)): return z_typ

#def AtomTyp(z_typ): 
#    if 'int' in str(type(z_typ)): return list(atomic_numbers.keys())[list(atomic_numbers.values()).index(z_typ)]
#    elif 'str' in str(type(z_typ)): return z_typ

def Create_P_matrix(pi_i,pi_f):
    P=np.zeros(shape=(len(pi_i),len(pi_i)))
    for i,j in zip(pi_i,pi_f):
        #P[i-1][j-1]=1 # base 1
        P[i][j] = 1 # Base 0
    return P

# ------ Special functions ---------
def SpecialFunctions(name_func, X, parameters):
    if name_func == 'func_h_bzn2graph':
        #print(name_func)
        return np.mean(X[parameters['benzene_C_idx']].T[2]) - np.mean(X[parameters['graphene_idx']].T[2])
    elif name_func == 'func_Angle_normal_bzn':
        mean_n = np.cross(X[parameters['benzene_C_idx'][2]] - X[parameters['benzene_C_idx'][0]],
                          X[parameters['benzene_C_idx'][4]] - X[parameters['benzene_C_idx'][0]])
        mean_n+= np.cross(X[parameters['benzene_C_idx'][3]] - X[parameters['benzene_C_idx'][1]],
                          X[parameters['benzene_C_idx'][5]] - X[parameters['benzene_C_idx'][1]])
        mean_n+= np.cross(X[parameters['benzene_C_idx'][4]] - X[parameters['benzene_C_idx'][2]],
                          X[parameters['benzene_C_idx'][0]] - X[parameters['benzene_C_idx'][2]])
        mean_n+= np.cross(X[parameters['benzene_C_idx'][5]] - X[parameters['benzene_C_idx'][3]],
                          X[parameters['benzene_C_idx'][1]] - X[parameters['benzene_C_idx'][3]])
        mean_n+= np.cross(X[parameters['benzene_C_idx'][0]] - X[parameters['benzene_C_idx'][4]],
                          X[parameters['benzene_C_idx'][2]] - X[parameters['benzene_C_idx'][4]])
        mean_n+= np.cross(X[parameters['benzene_C_idx'][1]] - X[parameters['benzene_C_idx'][5]],
                          X[parameters['benzene_C_idx'][3]] - X[parameters['benzene_C_idx'][5]])
        mean_n /= 6
        mean_n /= np.linalg.norm(mean_n)
        return np.degrees(np.arccos(np.dot(np.array([0, 0, 1]), mean_n)))

    elif name_func == 'func_HOMHED':
        # --- Reference structure -----------
        # n_atoms, typ, X0 = readxyz("salicylic.opt.xyz")
        # X0=X0.reshape(-1,3)
        # fnn=Comp_1nn(X0,Ring_Ph)
        #  {       "X-Y":  [Rs(Ang),Rd(Ang),Ropt(Ang),alpha] }
        HOMHED_param = {"CC": [1.530, 1.316, 1.387, 78.6],
                        "CN": [1.474, 1.271, 1.339, 87.4],
                        "NC": [1.474, 1.271, 1.339, 87.4],
                        "CO": [1.426, 1.210, 1.282, 77.2],
                        "OC": [1.426, 1.210, 1.282, 77.2],
                        "OH": [1.676, 0.933, 1.304, 7.22],  # Approx... we took the TS for the proton
                        "HO": [1.676, 0.933, 1.304, 7.22],
                        "NH": [1.550, 0.970, 1.260, 11.9],
                        "HN": [1.550, 0.970, 1.260, 11.9]}

        Ring_Ph = [1, 2, 4, 5, 6, 7]  # salicylic acid, Ph
        Ring_HB = [0, 1, 2, 3, 10, 8]  # salicylic acid, H-bond
        #HOMHED_ref(X, X0, fnn):
        #rij0 = scipy.spatial.distance.pdist(X0, 'euclidean')
        #rij0 = scipy.spatial.distance.squareform(rij0)
        #rij = scipy.spatial.distance.pdist(X, 'euclidean')
        #rij = scipy.spatial.distance.squareform(rij)
        suma = 0.
        #for ij in fnn:
            # print HOMHED_param[typ[ij[0]]+typ[ij[1]]],typ[ij[0]]+typ[ij[1]]
        #    suma += HOMHED_param[typ[ij[0]] + typ[ij[1]]][3] * (rij0[ij[0]][ij[1]] - rij[ij[0]][ij[1]]) ** 2
        return 0#1. - suma / float(len(fnn))

    elif name_func == 'func_ring':
        #
        #rij = scipy.spatial.distance.pdist(X, 'euclidean')
        #rij = scipy.spatial.distance.squareform(rij)
        return 0#[[index_i, index_j] for i, index_i in enumerate(ring) for j, index_j in enumerate(ring) if
                #rij[i][j] < 1.8 and rij[i][j] > 0. and j > i]

    elif name_func == 'func_':
        # =========== reaction coordinate =======
        # --- Function to compute:
        return 0#np.dot((a2 - a1).T, r - a1) / np.dot((a2 - a1).T, a2 - a1) - 0.5
    elif name_func == 'func_Force_dot_norm':
        # =========== F.n =====================
        #def funct_F_n(r, a2, F):
        return 0#np.dot(np.array(F), np.array(a2 - r) / np.linalg.norm(a2 - r)) * 1.e-3
    elif name_func == 'func_nn_ring':
        # =========== angle between rings =====================
        # def funct_nn_ring(r, label):
        #n1 = np.cross(r[label[0]] - r[label[1]], r[label[2]] - r[label[1]])
        #n1 /= np.linalg.norm(n1)
        #n2 = np.cross(r[label[3]] - r[label[4]], r[label[5]] - r[label[4]])
        #n2 /= np.linalg.norm(n2)
        # angle=np.degrees(np.arccos(min(1,max(np.dot(n1,n2),-1))))
        # if angle < 90.:
        #    return angle
        # else:
        #    return angle - 180.
        return None #np.degrees(np.arccos(min(1, max(np.dot(n1, n2), -1))))
    #elif name_func == 'func_':
    #elif name_func == 'func_':
    #elif name_func == 'func_':




# |********************************************|
# |***      END functions definitions       ***|
# |********************************************|


# |********************************************|
# |***      END functions definitions       ***|
# |********************************************|
 
# Read arguments.
if len(sys.argv) > 3:
    code = sys.argv[0]
    dataset_filename = sys.argv[1]
    Var1 = sys.argv[2]
    Var2 = sys.argv[3]
    if len(sys.argv) >= 4:
        use_data_range = False
        use_Symm = False
        Symm_str = ''
        every = 1
        Nmin = None
        Nmax = None
        bead = 0
        dt=0.001
        Random_True = False
        for ll in sys.argv:
            if "dt_ps=" in ll:
                dt = float(ll[6:])
            if "Min=" in ll:
                Nmin = int(ll[4:])
                use_data_range = True
            if "Max=" in ll:
                Nmax = int(ll[4:])
                use_data_range = True
            if "Every=" in ll:
                if "Random" in ll:
                    Random_True = True
                    Number_Random = int(ll.split(':')[1])
                else:
                    every = int(ll[6:])
                    use_data_range = True
            if "Symm=" in ll:
                file_P = ll[5:]
                use_Symm = True
            if "Bead=" in ll:
                if ll[5:] == "Centroid":
                    bead = ll[5:]
                else: 
                    bead = int(ll[5:])

else:
    message ='Usage: this_code.py <trajectory_file.npz/hdf5> <Var1_x> <Var2_y>'
    message+='\n <Optional:Min=i Max=i Every={i,Random:i} ; Symm="file.npz" ; dt_ps=f>'
    message+='\n Var Format: rij[i,j], angle[i,j,k], dihedral[i,j,k,l]'
    message+='\n if coordinates use: v1v2 plane[normal=[n1,n2,n3]:h_min,h_max] where v1v2 = xy, yz, xz'
    message+='\n In case of hdf5: <Optional:Bead={0,3,Centroid}'
    message+='\n In case of special functions: func_h_bzn2graph[idx], func_Angle_normal_bzn[idx]'
    sys.exit(message)
    #sys.exit('Usage: this_code.py <trajectory_file.npz/hdf5> <Var1_x> <Var2_y> <Optional:Min=n Max=m Every:every> <Optional:Symm="file.npz">\n Var Format: rij[i,j], angle[i,j,k], dihedral[i,j,k,l]')


# ============== NPZ file ================
if ".npz" in dataset_filename:
    print('Loading npz file...')
    npz_file = dict(np.load(dataset_filename, allow_pickle=True))
    print('Keys in the file:', npz_file.keys())
    if Nmin == None: Nmin = 0
    if Nmax == None: Nmax = len(npz_file['R'][0])
    if Random_True:
        Symm_str += ".RandomSampl-"+str(int(Number_Random/1000))+"k"
        mol_structures = npz_file['R'][0][np.random.randint(Nmin, high=Nmax, size=Number_Random)]
    else:
        # mol_structures = npz_file['R'][0][Nmin:Nmax:every]
        mol_structures = npz_file['R'][Nmin:Nmax:every] # by Ludi
    if 'z' in npz_file.keys():
        Z_flag = 'z'
        Typ = [AtomTyp(z) for z in npz_file[Z_flag]]
    else:
        Z_flag = 'typ'
        Typ = [AtomTyp(t) for t in npz_file[Z_flag]]
    print('Atoms in the molecule:',Typ)
    print(Z_flag)
    n_atoms = len(Typ)

# ============== hdf5 file ================
elif ".hdf5" in dataset_filename:
    # Load the database
    print('Loading hdf5 file...')
    database = h5py.File(dataset_filename, 'r', swmr=True, libver='latest')
    # Load structure data and convert to atoms
    structures = database['molecules']
    print("n_molecules.shape",structures.shape) # (2500000, 32, 1, 15, 6)

    # Load metadata
    n_replicas = structures.attrs['n_replicas']
    print("n_replicas",n_replicas)
    n_molecules = structures.attrs['n_molecules']
    n_atoms = structures.attrs['n_atoms'][0]
    atom_types = structures.attrs['atom_types'][0][0]
    entries = structures.attrs['entries']
    
    #for prop in database:
    #...     print(prop)
    #...
    #molecules  -> (20000, 32, 1, 12, 6)
    #properties -> (20000, 32, 1, 37) : 37 = 1 [Energy] + 3(12) [Forces]
    print('Loading data...') 
    if Nmin == None: Nmin = 0
    if Nmax == None: Nmax = entries 
    if bead == "Centroid":
        print("Using the CENTROID...")
        mol_structures = structures[Nmin:Nmax:every, :, 0, :n_atoms, :3]
        mol_structures = np.mean(mol_structures, axis=1, keepdims=True)
        dataset_filename=dataset_filename[:-5]+".CENTROIDhdf5"
    else: 
        mol_structures = structures[Nmin:Nmax:every, bead, 0, :n_atoms, :3] # Bohr, molecule = 0

    print(atom_types)
    Typ = list(map(AtomTyp,atom_types)) #[AtomTyp(z) for z in atom_types]
    print(Typ)

# ============== Symmetries ================
if use_Symm:
    print(100*'=')
    print('Using Higgins group to symmetrize the data:')
    print(100*'-')
    
    file_perm = np.load(file_P)
    #Permute = file_perm['perms'] - 1
    Permute = []
    for idx, P_pi in enumerate(file_perm['perms']):
        P_tmp = Create_P_matrix(range(len(Typ)), P_pi)
        #P_tmp = Create_P_matrix(file_perm['perms'][0],P_pi)
        if True:
            Permute.append(np.linalg.det(P_tmp) * P_tmp)
            if idx == 0: Symm_str += ".Symm-sgdml-sign"
        else:
            Permute.append(P_tmp)
            if idx == 0: Symm_str += ".Symm-sgdml"
    
    Permute = np.array(Permute)

    #[0,1,6,5,4,3,2,7,9,8,14,13,12,11,10, -1.]
    print(file_perm['perms'])
    print(100*'-')
else: 
    Permute = np.array([np.eye(len(Typ))])
rangelenPerm=range(len(Permute))

# ============== Creating labels ================
if 'rij' in Var1:
    atoms_index1 = list(map(int,Var1[4:-1].split(',')))
    print(atoms_index1)
    Var1 = Var1[:3]
    Label1 = 'r_'+Typ[atoms_index1[0]]+str(atoms_index1[0])+'.'
    Label1 += Typ[atoms_index1[1]]+str(atoms_index1[1])
elif 'angle' in Var1:
    atoms_index1 = list(map(int,Var1[6:-1].split(',')))
    Var1 = Var1[:5]
    Label1 = 'Angl_'+Typ[atoms_index1[0]]+str(atoms_index1[0])+'.'
    Label1 += Typ[atoms_index1[1]]+str(atoms_index1[1])+'.'
    Label1 += Typ[atoms_index1[2]]+str(atoms_index1[2])
elif 'dihedral' in Var1:
    print ('Var1 is a dihedral angle.')
    atoms_index1 = list(map(int,Var1[9:-1].split(',')))
    Var1 = Var1[:8]
    Label1 = 'Dihe_'+Typ[atoms_index1[0]]+str(atoms_index1[0])+'.'
    Label1 += Typ[atoms_index1[1]]+str(atoms_index1[1])+'.'
    Label1 += Typ[atoms_index1[2]]+str(atoms_index1[2])+'.'
    Label1 += Typ[atoms_index1[3]]+str(atoms_index1[3])
if 'xy' in Var1 or 'yz' in Var1 or 'xz' in Var1:
    # if xy, then the coordinates will integrate on z
    #Label1 = Var1[:1]
    atoms_index2 = None
if 'func_' in Var1:
    print('Var1 is {}.'.format(Var1))
    parameters = {}
    if 'func_h_bzn2graph' == Var1:
        atoms_index1 = [0, 1, 2, 3, 4, 5]
        parameters['benzene_C_idx'] = np.array(atoms_index1)
        parameters['graphene_idx'] = np.array(range(62)[12:])
        print(parameters['graphene_idx'])
        Var1 = 'func_h_bzn2graph'
        Label1 = 'func_h_bzn2graph'
    elif 'func_Angle_normal_bzn' == Var1:
        atoms_index1 = [0, 1, 2, 3, 4, 5]
        parameters['benzene_C_idx'] = np.array(atoms_index1)
        Var1 = 'func_Angle_normal_bzn'
        Label1 = 'func_Angle_normal_bzn'

if 'rij' in Var2:
    atoms_index2 = list(map(int, Var2[4:-1].split(',')))
    Var2 = Var2[:3]
    Label2 = 'r_'+Typ[atoms_index2[0]]+str(atoms_index2[0])+'.'
    Label2 += Typ[atoms_index2[1]]+str(atoms_index2[1])
elif 'angle' in Var2:
    atoms_index2 = list(map(int,Var2[6:-1].split(',')))
    Var2 = Var2[:5]
    Label2 = 'Angl_'+Typ[atoms_index2[0]]+str(atoms_index2[0])+'.'
    Label2 += Typ[atoms_index2[1]]+str(atoms_index2[1])+'.'
    Label2 += Typ[atoms_index2[2]]+str(atoms_index2[2])
elif 'dihedral' in Var2:
    print ('Var2 is a dihedral angle.')
    atoms_index2 = list(map(int,Var2[9:-1].split(',')))
    Var2 = Var2[:8]
    Label2 = 'Dihe_'+Typ[atoms_index2[0]]+str(atoms_index2[0])+'.'
    Label2 += Typ[atoms_index2[1]]+str(atoms_index2[1])+'.'
    Label2 += Typ[atoms_index2[2]]+str(atoms_index2[2])+'.'
    Label2 += Typ[atoms_index2[3]]+str(atoms_index2[3])
if 'plane' in Var2:
    # plane[normal=(n1,n2,n3):h_min,h_max]
    atoms_index2 = None
    print('normal=', Var2)
    normal = list(map(float, Var2[14:].split(':')[0][:-1].split(',')))
    print('normal=', normal)
    h_interval = list(map(float, Var2[:-1].split(':')[1].split(',')))
    Var2 = Var1[1:]
    Var1 = Var1[:1]
    idx_v1 = 0 if Var1 == 'x' else (1 if Var1 == 'y' else 2)
    idx_v2 = 0 if Var2 == 'x' else (1 if Var2 == 'y' else 2)
    idx_norm = 2 if (Var1+Var2 == 'xy' or Var1+Var2 == 'yx') else \
                (0 if (Var1+Var2 == 'yz' or Var1+Var2 == 'zy') else 1)
    Label1 = Var1 + '_plane-' + Var1+Var2 + '_h-' + str(h_interval[0])+"-"+str(h_interval[1])+"Ang"
    Label2 = Var2 + '_plane-' + Var1+Var2 + '_h-' + str(h_interval[0])+"-"+str(h_interval[1])+"Ang"
if 'func_' in Var2:
    print('Var1 is {}.'.format(Var2))
    parameters = {}
    if 'func_h_bzn2graph' == Var2:
        atoms_index2 = [0, 1, 2, 3, 4, 5]
        parameters['benzene_C_idx'] = np.array(atoms_index2)
        parameters['graphene_idx'] = np.array(range(62)[12:])
        Var2 = 'func_h_bzn2graph'
        Label2 = 'func_h_bzn2graph'
    elif 'func_Angle_normal_bzn' == Var2:
        atoms_index2 = [0, 1, 2, 3, 4, 5]
        parameters['benzene_C_idx'] = np.array(atoms_index2)
        Var2 = 'func_Angle_normal_bzn'
        Label2 = 'func_Angle_normal_bzn'

# ============== Plotting details ================
HistLog = True#True #
Phase1 = -360
Phase2 = -360.
Numbins = 100
# AxisRange = [True, -180, 180, -180, 180]
# AxisRange = [False, -180, 180, -180, 180] # by Ludi
# AxisRange = [False, 2.4, 3.8, 0, 180]
AxisRange = [False, 1.320, 1.530, 1.320, 1.530]
tick_spacing = [False, 60]
Norm_in_AxisRange = False

MinTime = float(Nmin)*dt # ps
MaxTime = float(Nmax)*dt # ps

# Units to plot
ExtraString = str(int(MinTime))+"-"+str(int(MaxTime))+"ps"
ExtraString += Symm_str

# ============== Check if the files exist ================
NPZ_files=[]
NPZ_files.append(dataset_filename[:-4] + ".TimeSerie.Var_" + Label1 + "." + ExtraString  + '.npz')
NPZ_files.append(dataset_filename[:-4] + ".TimeSerie.Var_" + Label2 + "." + ExtraString  + '.npz')
#Exist_NPZ = Path(NPZ_files[0]).exists() and Path(NPZ_files[1]).exists()
Exist_NPZ1 = os.path.isfile(NPZ_files[0]) 
Exist_NPZ2 = os.path.isfile(NPZ_files[1])

if Exist_NPZ1: print('File exist:\n',NPZ_files[0],)
if Exist_NPZ2: print('File exist:\n',NPZ_files[1],)

# ============== Generate trajectory of atoms (ASE) ================
X_traj = []
if (not Exist_NPZ1) or (not Exist_NPZ2):
    print('Construction of the trajectory using ASE.atoms ...')
    print(f'{mol_structures.shape=}')
    # num_parallel_jobs = 4
    # X_sub_trajs = np.array_split(npz_file['R'][0][Nmin:Nmax],num_parallel_jobs)
    # def read_parallel_data(R):
    #     X_sub_traj = []
    #     for x_i in tqdm(R):
    #         for i, typ_i in enumerate(Typ):
    #             if typ_i == 'D' or typ_i == 'T':
    #                 Typ[i] = 'H'
    #         X_traj.append(Atoms(Typ, positions=x_i))
    #     return X_sub_traj
    # from joblib import Parallel, delayed
    # X_traj = Parallel(n_jobs=num_parallel_jobs)(delayed(read_parallel_data)(sub_traj) for sub_traj in X_sub_trajs)
    # print([tmp.shape for tmp in X_traj])
    # exit()

    for x_i in tqdm(npz_file['R'][0][Nmin:Nmax]):
    # for x_i in npz_file['R'][Nmin:Nmax]: # by Ludi
        for i, typ_i in enumerate(Typ):
            if typ_i == 'D' or typ_i == 'T':
                Typ[i] = 'H'
        # print(f"{x_i=}")
        # exit()
        X_traj.append(Atoms(Typ, positions=x_i))

# ============== Starts the actual calculation ================
traj_var1 = []
traj_var2 = []
print(100 * '=')
print('Start computation loop...')
print(100 * '-')
#for frame_i in X_traj:

for frame_i in tqdm(X_traj, ncols=120):
    #print(frame_i)
    #print(frame_i.positions)
    for P in Permute: #Run on the permutations
        #print "Perm E -> :\n",Perm[0]
        #print Perm[p_ij]
        #P_Dyn1 = np.array([Permutes_ij(Perm[p_ij],Perm[p_ij][n_atoms]*P_Dyn0)])
        #frame_i.positions = Permutes_ij(P,abs(P[n_atoms])*frame_i.positions)
        #frame_i.positions = Permutes_ij(P,frame_i.positions)
        frame_i.positions = np.matmul(P,frame_i.positions)
        #print(frame_i.positions)
        # --- Rij
        if 'rij' in Var1 and (not Exist_NPZ1):
            traj_var1.append(frame_i.get_distance(atoms_index1[0],atoms_index1[1]))
        if 'rij' in Var2 and (not Exist_NPZ2):
            traj_var2.append(frame_i.get_distance(atoms_index2[0],atoms_index2[1]))
        # --- Angles
        if 'angle' in Var1 and (not Exist_NPZ1):
            traj_var1.append(frame_i.get_angle(atoms_index1[0],atoms_index1[1],atoms_index1[2]))
        if 'angle' in Var2 and (not Exist_NPZ2):
            traj_var2.append(frame_i.get_angle(atoms_index2[0],atoms_index2[1],atoms_index2[2]))
        # --- Dihedral
        if 'dihedral' in Var1 and (not Exist_NPZ1):
            traj_var1.append(frame_i.get_dihedral(atoms_index1[0],atoms_index1[1],atoms_index1[2],atoms_index1[3]))
            #print (frame_i.get_dihedral(atoms_index1[0],atoms_index1[1],atoms_index1[2],atoms_index1[3]))
        if 'dihedral' in Var2 and (not Exist_NPZ2):
            traj_var2.append(frame_i.get_dihedral(atoms_index2[0],atoms_index2[1],atoms_index2[2],atoms_index2[3]))
        # --- Plane
        #if ('x' in Var1 or 'y' in Var1 or 'z' in Var1) and (not Exist_NPZ1):
        #    r_tmp = frame_i.get_positions() #wrap=False, keyword='center')
        #    for r_i in r_tmp:
        #        if r_i[idx_norm] > h_interval[0] and r_i[idx_norm] < h_interval[1]:
        #            traj_var1.append(r_i[idx_v1])
        #            traj_var2.append(r_i[idx_v2])
            #print (frame_i.get_dihedral(atoms_index1[0],atoms_index1[1],atoms_index1[2],atoms_index1[3]))
        #if ('x' in Var2 or 'y' in Var2 or 'z' in Var2) and (not Exist_NPZ2):
        #    traj_var2.append(frame_i.get_dihedral(atoms_index2[0], atoms_index2[1], atoms_index2[2], atoms_index2[3]))
        # --- func_h_bzn2graph
        if 'func_h_bzn2graph' in Var1 and (not Exist_NPZ1):
            print(Var1, parameters['benzene_C_idx'])
            traj_var1.append(SpecialFunctions(Var1, frame_i.positions, parameters))
        if 'func_h_bzn2graph' in Var2 and (not Exist_NPZ2):
            traj_var2.append(SpecialFunctions(Var2, frame_i.positions, parameters))
        # --- func_Angle_normal_bzn
        if 'func_Angle_normal_bzn' in Var1 and (not Exist_NPZ1):
            traj_var1.append(SpecialFunctions(Var1, frame_i.positions, parameters))
        if 'func_Angle_normal_bzn' in Var2 and (not Exist_NPZ2):
            traj_var2.append(SpecialFunctions(Var2, frame_i.positions, parameters))

#print(traj_var1)
#print(traj_var2)

# ------ Time serie -------
if not Exist_NPZ1:
    # Creates dictionaries
    # Var 1 
    base_vars = {'Var':         traj_var1,\
             'Var_Name':        Label1,\
             'Time_step':       dt,\
             'Original_File':   dataset_filename}

    print("saving npz for "+Label1+"...")
    np.savez_compressed(NPZ_files[0], **base_vars)
else: 
    dir1 = np.load(NPZ_files[0])
    traj_var1 = dir1['Var']

if not Exist_NPZ2:
    # Var 2
    base_vars = {'Var':         traj_var2,\
             'Var_Name':        Label2,\
             'Time_step':       dt,\
             'Original_File':   dataset_filename}

    print("saving npz for "+Label2+"...")
    np.savez_compressed(NPZ_files[1], **base_vars)
else: 
    dir2 = np.load(NPZ_files[1])
    traj_var2 = dir2['Var']

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm

def PlotComp_CorrMapPhase_SAVE(Var1, Phase1, Var2, Phase2, String1,String2, NumberBins, xLabel, yLabel, grid,AxisRange=[False],tick_spacing = [False]):
    # Apply phase shift
    for i, var in enumerate(Var1):
        if var > 180:
            Var1[i] += Phase1#-180.

    for i, var in enumerate(Var2):
        if var > 180:
            Var2[i] += Phase2#-180.
    String=String1+"vs"+String2

    #plt.hist2d(Var1, Var2, bins=NumberBins)#, norm=LogNorm())
    if HistLog:
        histResults = plt.hist2d(Var1, Var2, bins=NumberBins, cmap="jet", norm=LogNorm(), density=True)
        #histResults = plt.hist2d(Var1, Var2, bins=[30,NumberBins], cmap="jet", norm=LogNorm())
    else:
        if AxisRange[0]:
            if Norm_in_AxisRange:
                #print("Var1.shape", Var1.shape)
                #mask = ((Var1 > AxisRange[1]) & (Var1 < AxisRange[2]))
                #aux1 = np.where(Var1[mask])[0]
                #aux1 = np.where(Var1[ Var1 > AxisRange[1] and Var1 < AxisRange[2] ])
                #print("Debug:len(aux1)", aux1.shape)
                #mask = ((Var2[aux1] > AxisRange[3]) & (Var2[aux1] < AxisRange[4]))
                #aux2 = np.where(Var2[aux1][mask])[0]
                #print("Debug:len(aux2)", aux2.shape)
                #num_in_range = len(aux2)
                #num_in_range = 978.0
                histResults = plt.hist2d(Var1, Var2, bins=NumberBins, cmap="jet",
                                         density=True,
                                         vmax=0.03,
                                         #cmin=0.1,
                                         range=np.array([(AxisRange[1], AxisRange[2]), (AxisRange[3], AxisRange[4])]))
            else:
                histResults = plt.hist2d(Var1, Var2, bins=NumberBins, cmap="jet", cmin=0.1, density=True,
                                         range=np.array([(AxisRange[1], AxisRange[2]), (AxisRange[3], AxisRange[4])]))
        else:
            histResults = plt.hist2d(Var1, Var2, bins=NumberBins, cmap="jet", density=True,)
            #histResults = plt.hist2d(Var1, Var2, bins=NumberBins, cmap="jet", vmax=44)
    print(histResults[0])
    print("Num max of points in a bin:", np.amax(histResults[0]))
    print(histResults[1])
    print(histResults[2])

    dataset_filename_ = dataset_filename.split("/")[:-1] + ["plotting"] + [dataset_filename.split("/")[-1]]
    dataset_filename_ = "/".join(dataset_filename_)


    # dataset_filename = "/".join(dataset_filename.split("/")[:-1] + ["plotting"]+[dataset_filename.split("/")[-1]])
    # dataset_filename = str(dataset_filename_)
    SaveResults1 = open(dataset_filename_[:-4] +'.'+Label1+'-'+Label2+'.Hist.dat', "w")
    SaveResults1.write("counts\n")
    for i in range(len(histResults[0])):
        for j in range(len(histResults[0][0])):
            SaveResults1.write(str(histResults[0][i][j]) + " ")
        SaveResults1.write("\n")
    SaveResults1.write("xedges\n")
    for i in range(len(histResults[1])):
        SaveResults1.write(str(histResults[1][i]) + " ")
    SaveResults1.write("\nyedges\n")
    for i in range(len(histResults[2])):
        SaveResults1.write(str(histResults[2][i]) + " ")
    SaveResults1.close()

    if AxisRange[0]:
        print("Using pre-defined bounds...")
        plt.axis(AxisRange[1:])
    if tick_spacing[0] and AxisRange[0]:
        plt.xticks(np.arange(AxisRange[1], AxisRange[2]+1, tick_spacing[1]))
        plt.yticks(np.arange(AxisRange[3], AxisRange[4]+1, tick_spacing[1]))
    # Plot 2D histogram using pcolor
    plt.colorbar()
    #plt.clim(1, 44);
    if grid:
        plt.grid()
    plt.xlabel(xLabel, fontsize=14)
    plt.ylabel(yLabel, fontsize=14)
    # plt.text(60, .025, r'T = '+Temp, fontsize=20)
    #plt.axis([-80, 80, 1.75, 3.5])
    plt.title(Label1+' vs '+Label2, fontsize=14)
    if HistLog:
        plt.savefig(dataset_filename_[:-4] + ".CorrMap." + Label1+'-'+Label2+ "." + ExtraString + '_log.pdf',
            bbox_inches='tight')
        print("saved as:", dataset_filename_[:-4] + ".CorrMap." + Label1+'-'+Label2 + "." + ExtraString + '_log.pdf')
    else:
        plt.savefig(dataset_filename_[:-4] + ".CorrMap." + Label1+'-'+Label2 + "." + ExtraString + '.pdf',
            bbox_inches='tight')
        print("saved as:", dataset_filename_[:-4] + ".CorrMap." + Label1+'-'+Label2 + "." + ExtraString + '.pdf')
    plt.clf()  # Clear the figure for the next loop
    plt.close()
    
    #prefix + ".CorrMap_" + String + "." + ExtraString + '.pdf'
    print( "done")
    # Plot 2D histogram using pcolor

# --- Plot
print('Plotting...')
PlotComp_CorrMapPhase_SAVE(traj_var1, Phase1, traj_var2, Phase2,
                              Label1,Label2, Numbins,
                              Label1, Label2,
                              True,AxisRange=AxisRange,tick_spacing=tick_spacing)


