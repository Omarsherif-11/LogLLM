#!/bin/bash
#SBATCH --job-name=train_mac
#SBATCH --output=train_mac.out
#SBATCH --error=train_mac.err
#SBATCH --partition=gpu_h100
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=300G
#SBATCH --time=72:00:00

source /pfs/data6/home/hu/hu_hu/hu_abdeom01/miniconda3/etc/profile.d/conda.sh
conda activate logllm
cd /pfs/data6/home/hu/hu_hu/hu_abdeom01/logllm_work

which python
python --version

python train_mac.py