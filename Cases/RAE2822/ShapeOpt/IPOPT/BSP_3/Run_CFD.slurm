#!/bin/bash
#SBATCH --time=24:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=20
#SBATCH --job-name="C5"
#SBATCH --partition=IllinoisComputes
#SBATCH --account=kaijames-ic
#SBATCH --output=FAILURE.ERR

module purge
module load anaconda/2023-Jul/3
module load openmpi/4.1.1-gcc-9.2.0


python3 RunOptimIP.py

#mpirun -np 60 SU2_CFD turb_CRMSN.cfg > log.CFD
