#!/bin/bash
#SBATCH --job-name=preprocess
#SBATCH --output=preprocess.out
#SBATCH --error=preprocess.err
#SBATCH --partition=highmem
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=1024G
#SBATCH --time=3-00:00:00

conda init
conda activate logllm
cd /pfs/data6/home/hu/hu_hu/hu_abdeom01/logllm_work/prepareData

python sliding_window.py --datasets bgl mac android windows 