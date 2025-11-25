#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --output=preprocess.out
#SBATCH --error=preprocess.err
#SBATCH --partition=cpu
#SBATCH --nodes=1
#SBATCH --ntasks=64
#SBATCH --mem=300G
#SBATCH --time=3-00:00:00

conda init
conda activate logllm
cd /pfs/data6/home/hu/hu_hu/hu_abdeom01/logllm_work/prepareData

python sliding_window.py --datasets bgl mac android windows 