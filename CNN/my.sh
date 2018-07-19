#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=mscdata01
#SBATCH --output=071718_subset1_noise100.out
date
source activate
conda activate tensorflow
module load usermods
module load user/cuda
python vgg16_mult_folder.py 0

date
conda deactivate
source deactivate