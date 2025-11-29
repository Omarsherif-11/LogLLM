#!/bin/bash
#SBATCH --job-name=train_android
#SBATCH --output=train_android.out
#SBATCH --error=train_android.err
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=64G
#SBATCH --time=10:00:00

source /pfs/data6/home/hu/hu_hu/hu_abdeom01/miniconda3/etc/profile.d/conda.sh
conda activate logllm
cd /pfs/data6/home/hu/hu_hu/hu_abdeom01/logllm_work

which python
python --version

python train_android.py