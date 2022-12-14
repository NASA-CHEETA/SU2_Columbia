#--- This script runs the dynamic and static forward analysis ad adjoint calculations for MDA
#--- Must be called for each design iteration during MDO

import os
import shutil
import glob
import wrtInp

#--- Get root directory 
Root = os.getcwd()

#--- Create an input deck for CCX for the first iteration only
# ---1 -Flag : Write input deck fr dynamic analysis 

if(n==1):
    for file in glob.glob('*.msh'):
        Flag = wrtInp.wrtInput(file, 1, E, nu, rho)
        if(Flag!=0):
            print("Error creating an input deck at iteration: ", n)
            


