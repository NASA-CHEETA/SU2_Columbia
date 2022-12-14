#!/usr/bin/env python

## \file nlopt_tools.py
#  \brief tools for interfacing with nlopt
#  \author Prateek. Ranjan
#  \version 7.2.0 "Blackbird"

import nlopt
import numpy as np
from numpy import * 
import sys
from .. import eval as su2eval

def nlopt_optim(project,x0,xb,its,accu):
    """ Runs nlopt implementation with an SU2 project

    Inputs:
            project - an SU2 project
            x0      - optional, initial guess
            xb      - optional, design variable bounds
            its     - max outer iterations, default 100
            accu    - accuracy, default 1e-10

    """



    #handle input cases
    #if x0 is None: x0 = []
    #if xb is None: xb = []




    # Number of design varibales
    dv_size = project.config['DEFINITION_DV']['SIZE']
    n_dv = sum(dv_size)
    project.n_dv = n_dv

    #Set the opt object definition
    optimizer = "LD_MMA"
    opt = nlopt.opt(nlopt.LD_MMA, n_dv)

    # Set the initial guess to zero
    if not x0: x0 = [0.0]*n_dv

    # Pre-scale the DV
    dv_scales = project.config['DEFINITION_DV']['SCALE']
    k = 0
    for i, dv_scl in enumerate(dv_scales):
        for j in range(dv_size[i]):
            x0[k] =x0[k]/dv_scl
            k = k + 1

    #Scale accuracy
    obj = project.config['OPT_OBJECTIVE']
    obj_scale = []
    for this_obj in obj.keys():
        obj_scale = obj_scale + [obj[this_obj]['SCALE']]
        
    #Scale accuracy
    eps = 1.0e-20


    #Get the lower and upper bounds
    lb = [0.0]*n_dv
    ub = [0.0]*n_dv
    for i in range(n_dv):
        lb[i] = np.array(xb[i][0])
        ub[i] = np.array(xb[i][1])

    #x0 = np.zeros(n_dv)

    print_summary (optimizer, n_dv, obj_scale, its, accu, x0, xb)

    #Define the objective function
    opt.set_min_objective(func)

    #Set bounds for the optimization problem
    opt.set_lower_bounds(lb)
    opt.set_upper_bounds(ub)

    #x0 = np.zeros(n_dv)
    
    xopt = opt.optimize(x0,project)
    

def print_summary(optimizer_name, n_dv, obj_scale, its, accu, x0, xb):
# optimizer summary
    sys.stdout.write(optimizer_name + ' parameters:\n')
    sys.stdout.write('Number of design variables: ' + str(n_dv) + '\n')
    sys.stdout.write('Objective function scaling factor: ' + str(obj_scale) + '\n')
    sys.stdout.write('Maximum number of iterations: ' + str(its) + '\n')
    sys.stdout.write('Requested accuracy: ' + str(accu) + '\n')
    sys.stdout.write('Initial guess for the independent variable(s): ' + str(x0) + '\n')
    sys.stdout.write('Lower and upper bound for each independent variable: ' + str(xb) + '\n\n')


def func(x, grad):
    f = 0
    obj_list = project.obj_f(x)
    for this_obj in obj_list:
        f = f + this_obj

    if grad.size > 0:
        dobj_list = project.obj_df(x)
        dobj=[0.0]*len(dobj_list[0])

    for this_dobj in dobj_list:
        idv=0
        for this_dv_dobj in this_dobj:
            dobj[idv] = dobj[idv]+this_dv_dobj
            idv+=1
        grad = array(dobj)

    return f   

        












