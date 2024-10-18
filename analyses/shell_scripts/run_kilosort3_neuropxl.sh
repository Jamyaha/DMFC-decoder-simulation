#!/usr/bin/env bash


#SBATCH -t 2-12:0:0
#SBATCH -N 1
#SBATCH --mem 120000
#SBATCH -p jazayeri
#SBATCH --gres=gpu:1

export MW_NVCC_PATH=/cm/shared/openmind/cuda/9.1/bin

module add openmind/cuda/9.1
module add openmind/cudnn/9.1-7.0.5
module add openmind/gcc/5.3.0
module add mit/matlab/2018b

NEW_UUID=$(cat /dev/urandom | tr -dc 'a-zA-Z0-9' | fold -w 32 | head -n 1)
tmp_fn=tmp_ks$NEW_UUID

# kilosort 1
#echo "addpath (\"/om/user/rishir/lib/KiloSort/\"); EntryPoint_rr(\"$@\"); exit;" > $tmp_fn.m
# kilosort 2
#echo "addpath (\"/om/user/rishir/lib/Kilosort2/\"); EntryPoint_rr(\"$@\"); exit;" > $tmp_fn.m
# kilosort 2.5
#echo "addpath (\"/om/user/rishir/lib/Kilosort_2.5/\"); EntryPoint_rr(\"$@\"); exit;" > $tmp_fn.m
# kilsort 3

echo "addpath (\"/om/user/rishir/lib/Kilosort/\");" > $tmp_fn.m
echo "EntryPoint_neuropxl(\"$@\", 1);" >> $tmp_fn.m
echo "EntryPoint_neuropxl(\"$@\", 2);" >> $tmp_fn.m
echo "EntryPoint_neuropxl(\"$@\", 3);" >> $tmp_fn.m
echo "EntryPoint_neuropxl(\"$@\", 4);" >> $tmp_fn.m
echo "exit;" >> $tmp_fn.m



matlab -nodisplay -r "$tmp_fn"
rm -f $tmp_fn.m



