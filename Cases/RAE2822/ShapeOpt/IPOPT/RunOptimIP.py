# Dummy script for FADO_DEV

from FADO import *
import subprocess
import numpy as np
import csv
import os
import glob
import shutil
import pandas as pd
import ipyopt
from numpy import ones, array, zeros

# Design Variables-----#
nDV = 24
x0 = np.zeros((nDV,))

# Define InputVariable class object: ffd
ffd = InputVariable(0.0, PreStringHandler("DV_VALUE="), nDV)
ffd = InputVariable(x0,ArrayLabelReplacer("__FFD_PTS__"), 0, np.ones(nDV), -0.01,0.01)

# Replace %__DIRECT__% with an empty string when using enable_direct
enable_direct = Parameter([""], LabelReplacer("%__DIRECT__"))

# Replace %__ADJOINT__% with an empty string when using enable_adjoint
enable_adjoint = Parameter([""], LabelReplacer("%__ADJOINT__"))

# Replace *_def.cfg with *_FFD.cfg as required
mesh_in = Parameter(["MESH_FILENAME= RAE2822_M00_FFD.su2"],\
       LabelReplacer("MESH_FILENAME= RAE2822_M00_FFD_def.su2"))

# Replace "OBJ_FUNC= SOME NAME" with "OBJ_FUNC= DRAG"
func_drag = Parameter(["OBJECTIVE_FUNCTION= DRAG"],\
         LabelReplacer("OBJECTIVE_FUNCTION= TOPOL_COMPLIANCE"))

# Replace "OBJ_FUNC= SOME NAME" with "OBJ_FUNC= MOMENT_Z"
func_mom = Parameter(["OBJECTIVE_FUNCTION= MOMENT_Z"],\
         LabelReplacer("OBJECTIVE_FUNCTION= TOPOL_COMPLIANCE"))

# EVALUATIONS---------------------------------------#

#Number of of available cores
ncores = "20"

# Master cfg file used for DIRECT and ADJOINT calculations
configMaster="FADO_RAE2822.cfg"

# cfg file used for GEOMETRY calculations
geoMaster = "RAE2822_geo.cfg"

# Input mesh to perform deformation
meshName="RAE2822_M00_FFD.su2"

# Mesh deformation
def_command = "mpirun -n " + ncores + " SU2_DEF " + configMaster

# Geometry evaluation
geo_command = "mpirun -n " + ncores + " SU2_GEO " + geoMaster

# Forward analysis
cfd_command = "mpirun -n " + ncores + " SU2_CFD " + configMaster

# Adjoint analysis -> AD + projection
adjoint_command = "mpirun -n " + ncores + " SU2_CFD_AD " + configMaster + " && mpirun -n " + ncores + " SU2_DOT_AD " + geoMaster

# Define the sequential steps for shape-optimization
max_tries = 1

# MESH DEFORMATON------------------------------------------------------#
deform = ExternalRun("DEFORM",def_command,True)
deform.setMaxTries(max_tries)
deform.addConfig(configMaster)
deform.addData("RAE2822_M00_FFD.su2")
deform.addExpected("RAE2822_M00_FFD_def.su2")
deform.addParameter(enable_direct)
deform.addParameter(mesh_in)

# GEOMETRY EVALUATION---------------------------------------------------#
geometry = ExternalRun("GEOMETRY",geo_command,True)
geometry.setMaxTries(max_tries)
geometry.addConfig(geoMaster)
geometry.addConfig(configMaster)
geometry.addData("DEFORM/RAE2822_M00_FFD_def.su2")
geometry.addExpected("of_func.csv")
geometry.addExpected("of_grad.csv")

# FORWARD ANALYSIS---------------------------------------------------#
direct = ExternalRun("DIRECT",cfd_command,True)
direct.setMaxTries(max_tries)
direct.addConfig(configMaster)
direct.addData("DEFORM/RAE2822_M00_FFD_def.su2")
direct.addExpected("solution.dat")
direct.addParameter(enable_direct)

# ADJOINT ANALYSIS---------------------------------------------------#
def makeAdjRun(name, func=None) :
    adj = ExternalRun(name,adjoint_command,True)
    adj.setMaxTries(max_tries)
    adj.addConfig(configMaster)
    adj.addConfig(geoMaster)
    adj.addData("DEFORM/RAE2822_M00_FFD_def.su2")
    adj.addData("DIRECT/solution.dat")
    adj.addData("DIRECT/flow.meta")
    adj.addExpected("of_grad.dat")
    adj.addParameter(enable_adjoint)
    if (func is not None) : adj.addParameter(func)
    return adj

# Drag adjoint
drag_adj = makeAdjRun("DRAG_ADJ",func_drag)

# Moment adjoint
mom_adj = makeAdjRun("MOM_ADJ",func_mom)

# FUNCTIONS ------------------------------------------------------------ #
drag = Function("drag","DIRECT/history.csv",LabeledTableReader('"CD"'))
drag.addInputVariable(ffd,"DRAG_ADJ/of_grad.dat",TableReader(None,0,(1,0)))
drag.addValueEvalStep(deform)
drag.addValueEvalStep(direct)
drag.addGradientEvalStep(drag_adj)
drag.setDefaultValue(1.0)

mom = Function("mom","DIRECT/history.csv",LabeledTableReader('"CMz"'))
mom.addInputVariable(ffd,"MOM_ADJ/of_grad.dat",TableReader(None,0,(1,0)))
mom.addValueEvalStep(deform)
mom.addValueEvalStep(direct)
mom.addGradientEvalStep(mom_adj)
mom.setDefaultValue(-1.0)

area = Function("a_min","GEOMETRY/of_func.csv",LabeledTableReader('"AIRFOIL_AREA"'))
area.addInputVariable(ffd,"GEOMETRY/of_grad.csv",LabeledTableReader('"AIRFOIL_AREA"',',',(0,None)))
area.addValueEvalStep(deform)
area.addValueEvalStep(geometry)
area.setDefaultValue(0.0)

# SCALING PARAMETERS ------------------------------------------------------------ #
GlobalScale = 1
ConScale = 1
FtolCr = 1E-12
Ftol = FtolCr * GlobalScale
OptIter = 5


# DRIVERL IPOPT-------------------------------------------------------#
driver = IpoptDriver()

# Objective function
driver.addObjective("min", drag, GlobalScale)

# Moment constraint
driver.addUpperBound(mom, 0.092, GlobalScale)

driver.addLowerBound(mom, 0.09199, GlobalScale)

# Area constraint
driver.addLowerBound(area, 0.0780934, GlobalScale)

nlp = driver.getNLP()

# ipyopt.set_loglevel(ipyopt.LOGGING_DEBUG)

# Optimization
x0 = driver.getInitial()

#x0 = array([3.73814470e-05, -1.25664179e-03, -1.43724701e-03, -1.40198147e-03,
# -1.21024455e-03, -1.19528048e-03, -1.76112075e-03,  5.89751808e-03,
#  8.67949004e-05, -8.13184641e-04, -4.23826233e-04,  2.64884393e-04,
#  2.07658957e-03, -3.49210062e-05, -1.32641431e-03,  5.52941576e-03,
#  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,
#  0.00000000e+00,  0.00000000e+00,  0.00000000e+00,  0.00000000e+00])

print("Initial Design Variable vector:")
print(x0)

# Warm start parameters
ncon = 3

#ubMult = array([1.33506674, 1.1747711,  1.15846737, 1.16227477, 1.19011096, 1.19600957,
# 1.13105525, 0.03668085, 1.34894454, 1.22115938, 1.26765997, 1.19816855,
# 1.69001066, 1.34019032, 1.17331481, 0.27101527, 1.31060325, 1.31060325,
# 1.31060325, 1.31060325, 1.31060325, 1.31060325, 1.31060325, 1.31060325,])

#lbMult = array([1.27720432, 1.48025902, 1.50522449, 1.49852626, 1.4379713,  1.4059509,
# 1.30525512, 0.72623611, 1.25142591, 1.41293871, 1.35446727, 1.31064884,
# 0.97017233, 1.14013989, 1.18209898, 0.74979554, 1.31060325, 1.31060325,
# 1.31060325, 1.31060325, 1.31060325, 1.31060325, 1.31060325, 1.31060325])

#conMult = array([-5.19927652])

lbMult = np.zeros(nDV)
ubMult= np.zeros(nDV)
conMult = np.zeros(ncon)


# NLP settings
nlp.set(warm_start_init_point = "no",
            nlp_scaling_method = "none",    # we are already doing some scaling
            accept_every_trial_step = "no", # can be used to force single ls per iteration
            limited_memory_max_history = 15,# the "L" in L-BFGS
            max_iter = OptIter,
            tol = Ftol,                     # this and max_iter are the main stopping criteria
            acceptable_iter = OptIter,
            acceptable_tol = Ftol,
            acceptable_obj_change_tol=1e-6, # Cauchy-type convergence over "acceptable_iter"
            dual_inf_tol=1e-07,             # Tolerance for optimality criteria
            mu_min = 1e-8,                  # for very low values (e-10) the problem "flip-flops"
            recalc_y_feas_tol = 0.1,
            output_file = 'ipopt_output.txt')        # helps converging the dual problem with L-BFGS

x, obj, status = nlp.solve(x0, mult_g = conMult, mult_x_L = lbMult, mult_x_U = ubMult)

# Print the optimized results---->

print("Primal variables solution")
print("x: ", x)

print("Bound multipliers solution: Lower bound")
print("lbMult: ", lbMult)

print("Bound multipliers solution: Upper bound")
print("ubMult: ", ubMult)


print("Constraint multipliers solution")
print("lambda:",conMult)



