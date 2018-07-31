#!/bin/sh
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --nodelist=mscdata01
#SBATCH --output=072418_vgg16_softmax.out
date
source activate
conda activate tensorflow
module load usermods
module load user/cuda
python vgg16_softmax.py 0

date
conda deactivate
source deactivate