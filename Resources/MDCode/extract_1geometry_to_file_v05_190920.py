#!/usr/bin/python

import sys
import numpy as np


from ase.data import atomic_numbers as _z_str_to_z_dict

Z_2_typ = {v: k for k, v in _z_str_to_z_dict.items()}

R_string = 'R'
if_Get_R_idx_rom_DataBaseFile = False
idx_string='idxs_valid'


# Read arguments.
if len(sys.argv) >= 4:
        model_filename = sys.argv[1]
        index = int(sys.argv[3])
        formato = sys.argv[2]
        if len(sys.argv) > 4: psi4_method = sys.argv[4]
        for ll in sys.argv:
            if 'R_string:' in ll:
                R_string = ll[9:]
            if 'idx_DataBaseFile:' in ll:
                Get_R_idx_from_DataBaseFile = ll[17:]
                if_Get_R_idx_rom_DataBaseFile = True
            if 'idx_DataBaseFile_label:' in ll:
                idx_string = ll[23:]
else:
    sys.exit('Usage: extract_1geometry.py <model_filename> <format: xyz, aims, psi4 or psi4-sapt> <geometrys_index> <if Psi4: method_file> <OPT R_string:s>')


# Load model.
try:
    if ".npz" in model_filename:
        model = dict(np.load(model_filename, allow_pickle=True))
        TrTe="train"
        model_filename+="_geometry_index:"+str(index)
    if if_Get_R_idx_rom_DataBaseFile:
        idx_file = dict(np.load(Get_R_idx_from_DataBaseFile, allow_pickle=True))
        print('old index:', index)
        index = idx_file[idx_string][index]
        print('new index:', index)
    if ".npy" in model_filename:
        model = np.load(model_filename).item()
        TrTe="test"
except:
        sys.exit("ERROR: Reading model file failed.", model_filename)

#for i in model:
#        print i
        #print model[i]
print(f"{model['R'].shape=}")
print(f"{R_string=}")
print(f"{index=}")
model = dict(model)
if model['R'].ndim==4: model.update({'R': model['R'].squeeze(0)})
elif model['R'].ndim>4: raise ValueError(f"Positional data model['R'] has more than [1, TimeSteps, Atoms, Dims] dimensions")

if R_string in model:
        x_idx = model[R_string][index]
        print("================= EXTRACTING training GEOMETRY:",index," =============")
        print("Number of atoms:",len(model[R_string][index]))
        Natoms = len(model[R_string][index])
        #print(model['R'][index])
        #print(model['z'])
        if 'z' in model:
               typ = [ Z_2_typ[z] for z in model['z'] ]
        else:
               typ = list(model['typ'])
        #print(typ)
        if formato == "aims":
               f = open("geometry.in", "w")
               f.write("#============================================\n#FHI-aims file\n#Model: "+model_filename)
               # If units of R in Bohr, then has to be changed to Ang using the FHIaims internal conversion constant
               # parameter ( bohr    = 0.52917721d0 )
               if 'r_unit' in model.keys():
                   if model['r_unit'] == 'bohr':
                       x_idx *= 0.52917721
                       f.write("\n#Original units bohr. Converted to Ang using FHIaims value: 1 bohr = 0.52917721 Ang.")
               f.write("\n#Training geometry index: "+str(index)+"\n#============================================")
               for i in range(Natoms):
                      f.write("\natom "+str(x_idx[i][0])+" "+str(x_idx[i][1])+" "+str(x_idx[i][2])+" "+typ[i])
               f.close()
        elif formato == "xyz":
                model_filename = "/".join(model_filename[:-4].split("/")[:-1] + ["plotting"] + [model_filename.split("/")[-1][:-4]])
                print(f"Saving to {model_filename}...")
                f = open(model_filename+"_geometry.xyz", "w")
                f.write(str(Natoms)+"\nModel:"+model_filename+",Training geometry index: "+str(index))
                for i in range(Natoms):
                        f.write("\n"+typ[i]+" "+str(x_idx[i][0])+" "+str(x_idx[i][1])+" "+str(x_idx[i][2]))
                f.close()
        elif formato == "psi4":
               f = open("geometry_psi4.in","w")
               f.write("#"+model_filename+" geometry_index_database:"+str(index)+" \nmolecule M {")
               if 'r_unit' in model.keys():
                   if model['r_unit'] == "bohr": f.write("\nunits bohr")
               for i in range(Natoms):
                      f.write("\n"+typ[i]+" "+str(x_idx[i][0])+" "+str(x_idx[i][1])+" "+str(x_idx[i][2]))
               f.write("\n}\n")
               g = open(psi4_method, "r")
               for line in g:
                      f.write(line)
               g.close()
               f.close()
        elif formato == "psi4-sapt":
               f = open("geometry_psi4.in", "w")
               f.write("#"+model_filename+", "+TrTe+" geometry index: "+str(index)+"\nmolecule M {\n")
               f.write("0 1")
               for i in range(model['n_atoms1']):
                      f.write("\n"+typ[i]+" "+str(x_idx[i][0])+" "+str(x_idx[i][1])+" "+str(x_idx[i][2]))
               f.write("\n--\n0 1")
               for i in range(model['n_atoms1'], model['n_atoms1']+model['n_atoms2']):
                      f.write("\n"+typ[i]+" "+str(x_idx[i][0])+" "+str(x_idx[i][1])+" "+str(x_idx[i][2]))
               #f.write("\n}\n")
               g=open(psi4_method, "r")
               for line in g:
                      f.write(line)
               g.close()
               f.close()
else:
        print('R not found.')
        print(model.files)


