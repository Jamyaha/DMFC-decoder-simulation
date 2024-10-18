#!/bin/sh


#SBATCH -t 6:0:0
#SBATCH -n 1
#SBATCH --mem-per-cpu 180000
#SBATCH --mail-type=NONE
#SBATCH --mail-user=rishir@mit.edu
##SBATCH --qos=jazayeri
##SBATCH --partition=jazayeri
#SBATCH --gres=gpu:0

#module add mit/matlab/2018a
NEW_UUID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
tmp_fn=tmp$NEW_UUID
echo "addpath (\"/om/user/rishir/lib/MentalPong/phys/matlab_scripts/\"); prepCleanOpenEphys_kilosort64(\"$@\"); exit;" > $tmp_fn.m
#echo "addpath (\"/om/user/rishir/lib/MentalPong/phys/matlab_scripts/\"); prepOpenEphys_kilosort64(\"$@\"); exit;" > $tmp_fn.m
echo "prepOpenEphys_kilosort64(\"$@\")"
module add mit/matlab/2018b
matlab -nodisplay -r "$tmp_fn"
rm -f $tmp_fn.m



