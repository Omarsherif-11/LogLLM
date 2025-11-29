#!/bin/bash
#SBATCH --job-name=train_mac
#SBATCH --output=train_mac.out
#SBATCH --error=train_mac.err
#SBATCH --partition=gpu_a100_il
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=1
#SBATCH --mem=32G
#SBATCH --time=05:00:00

source /pfs/data6/home/hu/hu_hu/hu_abdeom01/miniconda3/etc/profile.d/conda.sh
conda activate logllm
cd /pfs/data6/home/hu/hu_hu/hu_abdeom01/logllm_work

which python
python --version

python train_mac.py