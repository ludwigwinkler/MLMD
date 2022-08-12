#!/usr/bin/env bash

#!/bin/bash

if [ -d "IO" ]; then rm -rf IO; fi

#if [ "$#" -eq 2 ]; then
#  if ([ $1 = 'data' ] && [ $2 = 'model' ]) || ([ $2 = 'data' ] && [ $1 = 'model' ]); then
#    echo "Parallelizing over $1 and $2"
#    for data in keto_100K_0.2fs.npz keto_300K_0.2fs.npz keto_500K_0.2fs.npz; do
#      for model in lstm bi_lstm; do
#        qsub qsub_modelparallel_MLMD_Interpolation.sh $model $data
#      done
#    done
#  fi
#fi
#
#if [ "$#" -eq 1 ]; then
#  if [ $1 = 'data' ]; then
#    echo "Parallelizing over $1"
#    #  for data in benzene_dft.npz naphthalene_dft.npz ethanol_dft.npz malonaldehyde_dft.npz uracil_dft.npz paracetamol_dft.npz aspirin_dft.npz toluene_dft.npz salicylic_dft.npz; do
#  #  for data in keto_100K_0.2fs.npz keto_300K_0.2fs.npz keto_500K_0.2fs.npz; do
#  #    echo $data
#  #    qsub qsub_dataparallel_MLMD_Interpolation.sh $data
#  #  done
#  fi
#
#  if [ $1 = 'model' ]; then
#    echo "Parallelizing over $1"
#  #  for model in lstm ode hnn rnn bi_lstm bi_ode bi_hnn bi_rnn; do
#  #    qsub qsub_modelparallel_MLMD_Interpolation.sh $model
#  #    echo $model
#  #  done
#  fi
#fi
#
#if [ "$#" -eq 0 ]; then
#  echo "No parallelization arguments provided"
#fi

#for model in lstm rnn ode hnn bi_lstm bi_rnn bi_ode bi_hnn; do
for model in bi_lstm; do
#  for data in benzene_dft.npz naphthalene_dft.npz ethanol_dft.npz malonaldehyde_dft.npz uracil_dft.npz paracetamol_dft.npz aspirin_dft.npz toluene_dft.npz salicylic_dft.npz; do
#  for data in benzene_dft.npz naphthalene_dft.npz ethanol_dft.npz malonaldehyde_dft.npz uracil_dft.npz paracetamol_dft.npz aspirin_dft.npz toluene_dft.npz salicylic_dft.npz; do
    for data in keto_100K_0.2fs.npz keto_300K_0.2fs.npz keto_500K_0.2fs.npz; do
      for T in 5 10 20; do
#      for data in keto_100K_0.2fs.npz keto_300K_0.2fs.npz keto_500K_0.2fs.npz; do
        qsub qsub_modeltimeparallel_MLMD_Interpolation.sh $model $data $T
#        python ../MLMD_Interpolation.py -model $model -dataset $data -plot 0 -show 0
#        python ../MLMD_Interpolation.py -h
      done
#    done
  done
done


##for model in ode rnn lstm bi_ode bi_rnn bi_lstm
#for model in lstm bi_lstm ode bi_ode; do
##  for data in keto_100K_0.2fs.npz keto_300K_0.2fs.npz keto_500K_0.2fs.npz; do
##  for data in keto_300K_0.2fs.npz ethanol_dft.npz salicylic_dft.npz keto_100K_0.2fs.npz; do
#    qsub qsub_modelparallel_MLMD_Interpolation.sh $model
##  done
#done
