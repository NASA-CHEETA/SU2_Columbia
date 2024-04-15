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
nDV = 599
x0 = np.zeros((nDV,))

# Define InputVariable class object: ffd
ffd = InputVariable(0.0, PreStringHandler("DV_VALUE="), nDV)
ffd = InputVariable(x0,ArrayLabelReplacer("__FFD_PTS__"), 0, np.ones(nDV), -1.1,1.1)

# Replace %__DIRECT__% with an empty string when using enable_direct
enable_direct = Parameter([""], LabelReplacer("%__DIRECT__"))

# Replace %__ADJOINT__% with an empty string when using enable_adjoint
enable_adjoint = Parameter([""], LabelReplacer("%__ADJOINT__"))


#enable_not_def = Parameter([""], LabelReplacer("%__NOT_DEF__"))
#enable_def = Parameter([""], LabelReplacer("%__DEF__"))

# Replace *_def.cfg with *_FFD.cfg as required
mesh_in = Parameter(["MESH_FILENAME= CHEETA_TURB_M00_FFD.su2"],\
       LabelReplacer("MESH_FILENAME= CHEETA_TURB_M00_FFD_def.su2"))

# Replace "OBJ_FUNC= SOME NAME" with "OBJ_FUNC= DRAG"
func_drag = Parameter(["OBJECTIVE_FUNCTION= DRAG"],\
         LabelReplacer("OBJECTIVE_FUNCTION= TOPOL_COMPLIANCE"))

# Replace "OBJ_FUNC= SOME NAME" with "OBJ_FUNC= MOMENT_Z"
func_mom = Parameter(["OBJECTIVE_FUNCTION= MOMENT_Z"],\
         LabelReplacer("OBJECTIVE_FUNCTION= TOPOL_COMPLIANCE"))

# EVALUATIONS---------------------------------------#

#Number of of available cores
ncores = "256"

# Master cfg file used for DIRECT and ADJOINT calculations
configMaster="turb_CHEETA_FADO.cfg"

# cfg file used for FUSELAGE GEOMETRY calculations
geoMasterFuse = "CHEETA_Fuse_geo.cfg"

# cfg file used for WING GEOMETRY calculations
geoMasterWing = "CHEETA_Wing_geo.cfg"

# Input mesh to perform deformation
meshName="CHEETA_TURB_M00_FFD.su2"

# Mesh deformation
def_command = "mpirun -n " + ncores + " SU2_DEF " + configMaster

# Geometry evaluation
geo_commandFuse ="mpirun -n " + ncores + " SU2_GEO " + geoMasterFuse
geo_commandWing = "mpirun -n " + ncores +  " SU2_GEO " + geoMasterWing

# Forward analysis
cfd_command = "mpirun -n " + ncores + " SU2_CFD " + configMaster

# Adjoint analysis -> AD + projection
adjoint_command = "mpirun -n " + ncores + " SU2_CFD_AD " + configMaster + " && mpirun -n " + ncores + " SU2_DOT_AD " + geoMasterFuse

# Define the sequential steps for shape-optimization
max_tries = 1

# MESH DEFORMATON------------------------------------------------------#
deform = ExternalRun("DEFORM",def_command,True)
deform.setMaxTries(max_tries)
deform.addConfig(configMaster)
deform.addData("CHEETA_TURB_M00_FFD.su2")
deform.addExpected("CHEETA_TURB_M00_FFD_def.su2")
deform.addParameter(enable_direct)
deform.addParameter(mesh_in)
#deform.addParameter(enable_def)

# GEOMETRY EVALUATION FUSELAGE-----------------------------------------#
geometryF = ExternalRun("GEOMETRY_FUSE",geo_commandFuse,True)
geometryF.setMaxTries(max_tries)
geometryF.addConfig(geoMasterFuse)
geometryF.addConfig(configMaster)
geometryF.addData("DEFORM/CHEETA_TURB_M00_FFD_def.su2")
geometryF.addExpected("of_func.csv")
geometryF.addExpected("of_grad.csv")

# GEOMETRY EVALUATION WING-----------------------------------------#
geometryW = ExternalRun("GEOMETRY_WING",geo_commandWing,True)
geometryW.setMaxTries(max_tries)
geometryW.addConfig(geoMasterWing)
geometryW.addConfig(configMaster)
geometryW.addData("DEFORM/CHEETA_TURB_M00_FFD_def.su2")
geometryW.addExpected("of_func.csv")
geometryW.addExpected("of_grad.csv")

# FORWARD ANALYSIS---------------------------------------------------#
direct = ExternalRun("DIRECT",cfd_command,True)
direct.setMaxTries(max_tries)
direct.addConfig(configMaster)
direct.addData("DEFORM/CHEETA_TURB_M00_FFD_def.su2")
direct.addExpected("solution.dat")
direct.addParameter(enable_direct)


# ADJOINT ANALYSIS---------------------------------------------------#
def makeAdjRun(name, func=None) :
    adj = ExternalRun(name,adjoint_command,True)
    adj.setMaxTries(max_tries)
    adj.addConfig(configMaster)
    adj.addConfig(geoMasterFuse)
    adj.addData("DEFORM/CHEETA_TURB_M00_FFD_def.su2")
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

mom = Function("mom","DIRECT/history.csv",LabeledTableReader('"CMy"'))
mom.addInputVariable(ffd,"MOM_ADJ/of_grad.dat",TableReader(None,0,(1,0)))
mom.addValueEvalStep(deform)
mom.addValueEvalStep(direct)
mom.addGradientEvalStep(mom_adj)
mom.setDefaultValue(-1.0)

FuseVol = Function("FuseVol","GEOMETRY_FUSE/of_func.csv",LabeledTableReader('"FUSELAGE_VOLUME"'))
FuseVol.addInputVariable(ffd,"GEOMETRY_FUSE/of_grad.csv",LabeledTableReader('"FUSELAGE_VOLUME"',',',(0,None)))
FuseVol.addValueEvalStep(deform)
FuseVol.addValueEvalStep(geometryF)
FuseVol.setDefaultValue(0.0)

WingVol = Function("WingVol","GEOMETRY_WING/of_func.csv",LabeledTableReader('"WING_VOLUME"'))
WingVol.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"WING_VOLUME"',',',(0,None)))
WingVol.addValueEvalStep(deform)
WingVol.addValueEvalStep(geometryW)
WingVol.setDefaultValue(0.0)

ST1_TH = Function("ST1_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION1_THICKNESS"'))
ST1_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION1_THICKNESS"',',',(0,None)))
ST1_TH.addValueEvalStep(deform)
ST1_TH.addValueEvalStep(geometryW)
ST1_TH.setDefaultValue(0.0)

ST2_TH = Function("ST2_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION2_THICKNESS"'))
ST2_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION2_THICKNESS"',',',(0,None)))
ST2_TH.addValueEvalStep(deform)
ST2_TH.addValueEvalStep(geometryW)
ST2_TH.setDefaultValue(0.0)

ST3_TH = Function("ST3_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION3_THICKNESS"'))
ST3_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION3_THICKNESS"',',',(0,None)))
ST3_TH.addValueEvalStep(deform)
ST3_TH.addValueEvalStep(geometryW)
ST3_TH.setDefaultValue(0.0)

ST4_TH = Function("ST4_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION4_THICKNESS"'))
ST4_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION4_THICKNESS"',',',(0,None)))
ST4_TH.addValueEvalStep(deform)
ST4_TH.addValueEvalStep(geometryW)
ST4_TH.setDefaultValue(0.0)

ST5_TH = Function("ST5_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION5_THICKNESS"'))
ST5_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION5_THICKNESS"',',',(0,None)))
ST5_TH.addValueEvalStep(deform)
ST5_TH.addValueEvalStep(geometryW)
ST5_TH.setDefaultValue(0.0)

ST6_TH = Function("ST6_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION6_THICKNESS"'))
ST6_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION6_THICKNESS"',',',(0,None)))
ST6_TH.addValueEvalStep(deform)
ST6_TH.addValueEvalStep(geometryW)
ST6_TH.setDefaultValue(0.0)

ST7_TH = Function("ST7_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION7_THICKNESS"'))
ST7_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION7_THICKNESS"',',',(0,None)))
ST7_TH.addValueEvalStep(deform)
ST7_TH.addValueEvalStep(geometryW)
ST7_TH.setDefaultValue(0.0)

ST8_TH = Function("ST8_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION8_THICKNESS"'))
ST8_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION8_THICKNESS"',',',(0,None)))
ST8_TH.addValueEvalStep(deform)
ST8_TH.addValueEvalStep(geometryW)
ST8_TH.setDefaultValue(0.0)

ST9_TH = Function("ST9_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION9_THICKNESS"'))
ST9_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION9_THICKNESS"',',',(0,None)))
ST9_TH.addValueEvalStep(deform)
ST9_TH.addValueEvalStep(geometryW)
ST9_TH.setDefaultValue(0.0)

ST10_TH = Function("ST10_TH","GEOMETRY_WING/of_func.csv",LabeledTableReader('"STATION10_THICKNESS"'))
ST10_TH.addInputVariable(ffd,"GEOMETRY_WING/of_grad.csv",LabeledTableReader('"STATION10_THICKNESS"',',',(0,None)))
ST10_TH.addValueEvalStep(deform)
ST10_TH.addValueEvalStep(geometryW)
ST10_TH.setDefaultValue(0.0)

# SCALING PARAMETERS ------------------------------------------------------------ #
GlobalScale = 1
ConScale = 1
FtolCr = 1E-12
Ftol = FtolCr * GlobalScale
OptIter = 2

# FUSELAGE AND WING VOLUME SCALING PARAMETERS (MKIII)
BSL_FUSE_VOL = 469.483
FACTOR_FV = 1
TRG_FUSE_VOL = BSL_FUSE_VOL * FACTOR_FV

BSL_WING_VOL = 15.1923
FACTOR_WV = 1
TRG_WING_VOL = BSL_WING_VOL * FACTOR_WV

# THICKNESS BOUND (xx % of BSL value)
THK_BND = 0.99

# Spanwise thickness values
ST1_T = 0.718022

ST2_T = 0.584234

ST3_T = 0.461259

ST4_T = 0.369912

ST5_T = 0.329719

ST6_T = 0.293053

ST7_T = 0.258708

ST8_T = 0.226926

ST9_T = 0.196369

ST10_T = 0.0953453


# DRIVER IPOPT-------------------------------------------------------#
driver = IpoptDriver()


 # Objective function
driver.addObjective("min", drag, GlobalScale)

nlp = driver.getNLP()

# Wing pitching moment constraint
#driver.addLowerBound(mom, -0.006543, GlobalScale , ConScale)

# Fuselage volume constraint
driver.addLowerBound(FuseVol, TRG_FUSE_VOL, GlobalScale , ConScale)

# Wing volume constraint
driver.addLowerBound(WingVol, TRG_WING_VOL, GlobalScale , ConScale)

# Span-wise thickness constraints
driver.addLowerBound(ST1_TH, ST1_T* THK_BND, GlobalScale , ConScale)
driver.addLowerBound(ST2_TH, ST2_T* THK_BND, GlobalScale , ConScale)
driver.addLowerBound(ST3_TH, ST3_T* THK_BND, GlobalScale , ConScale)
driver.addLowerBound(ST4_TH, ST4_T* THK_BND, GlobalScale , ConScale)
driver.addLowerBound(ST5_TH, ST5_T* THK_BND, GlobalScale , ConScale)
driver.addLowerBound(ST6_TH, ST6_T* THK_BND, GlobalScale , ConScale)
driver.addLowerBound(ST7_TH, ST7_T* THK_BND, GlobalScale , ConScale)
driver.addLowerBound(ST8_TH, ST8_T* THK_BND, GlobalScale , ConScale)
driver.addLowerBound(ST9_TH, ST9_T* THK_BND, GlobalScale , ConScale)
driver.addLowerBound(ST10_TH, ST10_T* THK_BND, GlobalScale , ConScale)


driver.setWorkingDirectory("WORKDIR")
driver.setEvaluationMode(False,2.0)

driver.setStorageMode(True,"DSN_")
driver.setFailureMode("HARD")

# Optimization
x0 = driver.getInitial()

# Warm start parameters
ncon = 12
lbMult = np.zeros(nDV)
ubMult= np.zeros(nDV)
conMult = np.zeros(ncon)

# WARM START PARAMETERS
#x0 = array([])

print("Initial Design Variable vector:")
print(x0)


#ubMult = array([])

#lbMult = array([])

#conMult = array([])



# NLP settings
nlp.set(warm_start_init_point = "no",
            nlp_scaling_method = "none",    # we are already doing some scaling
            accept_every_trial_step = "no", # can be used to force single ls per iteration
            limited_memory_max_history = 15,# the "L" in L-BFGS
            max_iter = OptIter,
            tol = Ftol,                     # this and max_iter are the main stopping criteria
            acceptable_iter = OptIter,
            acceptable_tol = Ftol,
            acceptable_obj_change_tol=1e-12, # Cauchy-type convergence over "acceptable_iter"
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
