#!/bin/bash
#SBATCH --time=4:00:00
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name="CRM_OP"
#SBATCH --partition=IllinoisComputes
#SBATCH --account=kaijames-ic
#SBATCH --output=FAILURE.ERR

module purge
module load anaconda/2023-Jul/3
module load openmpi/4.1.1-gcc-9.2.0

python3 RunOptimIP.py

#mpirun -np 40 SU2_CFD turb_CRMWS.cfg > log.CFD
