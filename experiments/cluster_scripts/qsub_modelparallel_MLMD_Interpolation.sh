#!/usr/bin/env bash
#$ -l cuda=1   # request one GPU (remove this line if none is needed)
#$ -l h=!(node28|node01)   # Do not place job on node28 which doesn't support neweset cuda
#$ -q all.q    # don't fill the qlogin queue
#$ -cwd        # change working directory (to current)
#$ -V
#$ -e IO/error.txt
#$ -o IO/output.txt

wandb login --relogin afc4755e33dfa171a8419620e141ebeaeb8f27f5

#for pct in 0.001 0.01 0.1 0.25 0.5 1.; do
#for pct in 1.; do
#  for data in benzene_dft.npz naphthalene_dft.npz keto_300K_0.2fs.npz ethanol_dft.npz salicylic_dft.npz keto_100K_0.2fs.npz; do
#  for data in benzene_dft.npz naphthalene_dft.npz ethanol_dft.npz malonaldehyde_dft.npz uracil_dft.npz paracetamol_dft.npz aspirin_dft.npz toluene_dft.npz salicylic_dft.npz; do
#  for data in ethanol_dft.npz ; do
#    for output_length_train in 10 20; do
      for output_length in 1 3 5 ; do
        for integration_mode in int diffeq ; do
          for diffeq_output_scaling in True False ; do
            python ../MLMD_Interpolation.py -logger True -plot True -show False\
              -model $1 -dataset ps_matrix -load_weights False\
              -input_length 10 -output_length_train $output_length -output_length_val $output_length\
              -max_epochs 2000 -fast_dev_run False -num_layers 5 -num_hidden_multiplier 20 -batch_size 50\
              -integration_mode $integration_mode -diffeq_output_scaling $diffeq_output_scaling
          done
        done
      done
#    done
#  done
#done
