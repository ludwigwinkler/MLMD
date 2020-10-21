#!/usr/bin/env bash
#$ -l cuda=1   # request one GPU (remove this line if none is needed)
#$ -l h=!(node28|node01)   # Do not place job on node28 which doesn't support neweset cuda
#$ -q all.q    # don't fill the qlogin queue
#$ -cwd        # change working directory (to current)
#$ -V
#$ -e IO/error.txt
#$ -o IO/output.txt

# test 123

wandb login --relogin afc4755e33dfa171a8419620e141ebeaeb8f27f5

#for pct in 0.001 0.01 0.1 0.25 0.5 1.; do
for pct in 1.; do
  for data in keto_100K_0.2fs.npz keto_300K_0.2fs.npz keto_500K_0.2fs.npz; do
    for output_length_train in 10 20 30; do
      for output_length_val in 10 20 30; do
        python MLMD_Interpolation.py -logger True -plot True -show False \
          -pct_data_set $pct -model $1 -data_set $data \
          -input_length 1 -output_length_train $output_length_train -output_length_val $output_length_val
      done
    done
  done
done
