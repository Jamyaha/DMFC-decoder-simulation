#!/usr/bin/env bash

#SBATCH -t 12:0:0
#SBATCH -n 1
#SBATCH --mem-per-cpu 60000
#SBATCH --mail-type=NONE
#SBATCH --mail-user=rishir@mit.edu


source ~/.bashrc
source activate py36
export PYTHONPATH=/home/rishir/envs/py36/lib/python3.6/site-packages/

python /om/user/rishir/lib/MentalPong/phys/run_scripts/DecodeOverTime.py "$@"