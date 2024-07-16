#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=2
#SBATCH --ntasks-per-node=128
#SBATCH --job-name="A1"
#SBATCH --partition=IllinoisComputes
#SBATCH --account=kaijames-ic
#SBATCH --output=FAILURE.ERR
#SBATCH --exclude=ccc0292
module purge
module load anaconda/2023-Jul/3
module load openmpi/4.1.1-gcc-9.2.0


python3 RunOptimIP.py


