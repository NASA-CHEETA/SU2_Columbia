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

# MOVE THESE TO FADO ROOT---------------------#
def getLatestIter():
    Conv = pd.read_csv('convergence.csv', skiprows = 0)
    Arr = Conv.to_numpy()
    Major_Iter = Arr[:,0]
    Last_Iter = int(np.max(Major_Iter))
    return Last_Iter

def restart(meshName = None, majIter = 0, minIter = 0):
    # STEP 1: Move the baseline to preserve file
    # Get path at root
    Root = os.getcwd()  
    if os.path.exists("BASELINE_MESH") and os.path.isdir("BASELINE_MESH"):
        print("Baseline mesh directory exists!")
    else:
        print("Moving baseline mesh")
        os.mkdir("BASELINE_MESH")
        for file in glob.glob('BASELINE_MESH'):
            # Move to this directory and get the path
            os.chdir(file)
            BSL_MSH = os.getcwd()
        # Preserve baseline mesh by moving it to "BASELINE_MESH"
        os.chdir(Root)
        for file in glob.glob("*.su2"):
            shutil.move(file, BSL_MSH)
            print("Done!")

    # STEP 2: Query the convergence file to get intermediate mesh
    Conv = pd.read_csv('convergence.csv', skiprows = 0)
    Arr = Conv.to_numpy()
    # Last optimization iteration

    Major_Iter = Arr[:,0]
    Last_Iter = int(np.max(Major_Iter))
    # Last function evaluation at the corresponding optimization iteration
    # Do Last_Iter -1 since python3 starts from 0
    Last_FEval = minIter
    Target_DIR = "DSN_{:03d}".format(Last_FEval)
    if os.path.isdir(Target_DIR):
        print("Original restart directory found.")
        shutil.rmtree("WORKDIR")
        os.chdir(str(Target_DIR)+"/DEFORM")
        for file in glob.glob("*FFD_def.su2"):
            shutil.copy(file, Root)
        os.chdir(Root)
        for file in glob.glob("*FFD_def.su2"):
            os.rename(file, meshName)
    else:
        print("Target directory does not exist but solution has converged")
        print("Renaming WORKDIR to ", Target_DIR)
        os.rename("WORKDIR", Target_DIR)
        # Now go into the target directory and move the mesh to root
        os.chdir(str(Target_DIR)+"/DEFORM")
        for file in glob.glob("*FFD_def.su2"):
            shutil.copy(file, Root)
        os.chdir(Root)
        for file in glob.glob("*FFD_def.su2"):
            os.rename(file, meshName)

    return Last_FEval
    

def initialize_file(filename):
    """
    Initializes the CSV file with the header.
    :param filename: Name of the file to initialize.
    """
    #header = ['ITER', 'FEVAL', 'OBJ', 'OPTIMALITY', 'CMY FSB', 'CMY JAC NORM', 'AREA FSB', 'AREA JAC NORM']
    header = ['ITER', 'FEVAL', 'OBJ', 'OPTIMALITY', 'MOM FSB', 'AREA FSB']

    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter=',', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(header)

    # Define a callback function
def store_data(xk, filename):
    """
    Queries variour values from driver class and writes them to
    a csv file.
    """
    global iteration

    # Query objective function value and optimality 
    objective_value = driver.funRec(xk)                                          
    optimality = np.linalg.norm(driver.grad(xk),ord=np.inf)    
                             
    # Query moment constraint feasibility
    feasb_mom = driver._eval_g(xk,0)                                          
    #mom_jacobian = np.linalg.norm(driver._eval_jac_g(xk,0),ord=np.inf)    

    # Query area constraint feasibility
    feasb_area = driver._eval_g(xk,1)                                          
    #area_jacobian = np.linalg.norm(driver._eval_jac_g(xk,1),ord=np.inf)      

    # Get counter for feval
    counter = driver.fEvalCtr()        
    data = [iteration, counter, objective_value, optimality, feasb_mom, feasb_area]
    #data = [iteration, counter, objective_value, optimality, feasb_mom, mom_jacobian,feasb_area,area_jacobian]
    formatted_data = [str(data[0]), str(data[1])] + ["{:.6e}".format(value) for value in data[2:]]


    iteration += 1

   
    with open(filename, 'a', newline='') as csvfile:
        writer = csv.writer(csvfile, delimiter = ',')
        writer.writerow(formatted_data)



# OPTIMZIATION RESTART SETTINGS---------------#
Restart = False
majIter = 0
minIter = 0

# GLobal variable to store the iteration value
if Restart:
    # Get the latest optimization iteration from the csv file
    iteration = getLatestIter() + 1
else:
    # Starting a new run, set starting iteration to 1
    iteration = 1

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
ncores = "8"

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
# driver.addUpperBound(mom, 0.092, GlobalScale)

# Area constraint
driver.addLowerBound(area, 0.0780934, GlobalScale)

nlp = driver.getNLP()

# ipyopt.set_loglevel(ipyopt.LOGGING_DEBUG)

# Optimization
x0 = driver.getInitial()

print("Initial Design Variable vector:")
print(x0)

# Warm start parameters
ncon = 1
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
print("lbMult: ", ubMult)


print("Constraint multipliers solution")
print("lambda:",conMult)



