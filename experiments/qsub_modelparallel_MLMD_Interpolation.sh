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
for pct in 1.; do
  for data in keto_300K_0.2fs.npz ethanol_dft.npz salicylic_dft.npz keto_100K_0.2fs.npz; do
    for output_length_train in 20; do
      for output_length_val in 20; do
#      echo $data
        python MLMD_Interpolation.py -logger True -plot True -show False\
          -pct_dataset $pct -model $1 -dataset $data -load_pretrained False\
          -input_length 3 -output_length_train $output_length_train -output_length_val $output_length_val
      done
    done
  done
done
