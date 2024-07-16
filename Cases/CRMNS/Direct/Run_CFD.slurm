#!/bin/bash
#SBATCH --time=10:00:00
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=20
#SBATCH --job-name="10_3"
#SBATCH --partition=IllinoisComputes
#SBATCH --account=kaijames-ic
#SBATCH --output=FAILURE.ERR

module purge
module load openmpi/4.1.1-gcc-9.2.0

mpirun -np 60 SU2_CFD turb_CRMSN.cfg > log.CFD
