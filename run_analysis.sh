#!/bin/bash

#SBATCH -p cpu
#SBATCH -N 1
#SBATCH -n 10
#SBATCH --mem 5000
#SBATCH -t 1-0:0
#SBATCH	-o analysis.out
#SBATCH -e analysis.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=federicoclaudi@protonmail.com

echo "Loading python environemnt"
module load miniconda
conda activate fp

echo "running"
python analysis/analyse_folder.py
