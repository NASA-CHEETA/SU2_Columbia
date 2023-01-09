#!/usr/bin/env python3

## \file pyoptsparse_tools.py
#  \brief tools for interfacing with pyoptsparse
#  \author R. Ranjan
#  \version 7.2.0 "Blackbird"
#
# SU2 Original Developers: Dr. Francisco D. Palacios.
#                          Dr. Thomas D. Economon.
#
# SU2 Developers: Prof. Juan J. Alonso's group at Stanford University.
#                 Prof. Piero Colonna's group at Delft University of Technology.
#                 Prof. Nicolas R. Gauger's group at Kaiserslautern University of Technology.
#                 Prof. Alberto Guardone's group at Polytechnic University of Milan.
#                 Prof. Rafael Palacios' group at Imperial College London.
#                 Prof. Edwin van der Weide's group at the University of Twente.
#                 Prof. Vincent Terrapon's group at the University of Liege.
#
# SU2 MDO Developers : Prateek Ranjan, University of Illinois at Urbana-Champaign
#
# Copyright (C) 2012-2017 SU2, the open-source CFD code.
#
# SU2 is free software; you can redistribute it and/or
# modify it under the terms of the GNU Lesser General Public
# License as published by the Free Software Foundation; either
# version 2.1 of the License, or (at your option) any later version.
#
# SU2 is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
# Lesser General Public License for more details.
#
# You should have received a copy of the GNU Lesser General Public
# License along with SU2. If not, see <http://www.gnu.org/licenses/>.

# -------------------------------------------------------------------
#  Imports
# -------------------------------------------------------------------

import sys

from .. import eval as su2eval
from numpy import array, zeros, concatenate

import pyoptsparse
#from pyoptsparse import Optimization

#from pyoptsparse import SLSQP, Optimziation

from pyoptsparse import SLSQP, PSQP



def pyOptSparse_optimization(project,x0=None,xb=None,its=100,accu=1e-10, optimizer = 'SLSQP'):
    print("Running pyOptSparse ....")

    """ Runs the pyOptSparse implementation of SLSQP with
        an SU2 project

        Inputs:
            project - an SU2 project
            x0      - optional, initial guess
            xb      - optional, design variable bounds
            its     - max outer iterations, default 100
            accu    - accuracy, default 1e-10

        Outputs:
           result - the outputs from pyoptsparse
    """

    optOptions = {}

    # handle input cases
    if x0 is None: x0 = []
    if xb is None: xb = []

    # number of design variables
    dv_size = project.config['DEFINITION_DV']['SIZE']
    n_dv = sum( dv_size)
    project.n_dv = n_dv

    # Initial guess
    if not x0: x0 = [0.0]*n_dv

    # prescale x0
    dv_scales = project.config['DEFINITION_DV']['SCALE']
    k = 0
    for i, dv_scl in enumerate(dv_scales):
        for j in range(dv_size[i]):
            x0[k] =x0[k]/dv_scl
            k = k + 1

    # scale accuracy
    obj = project.config['OPT_OBJECTIVE']
    obj_scale = []
    for this_obj in obj.keys():
        obj_scale = obj_scale + [obj[this_obj]['SCALE']]

    # scale accuracy
    eps = 1.0e-20

    lb = [0.0]*n_dv
    ub = [0.0]*n_dv
    for i in range(n_dv):
        lb[i] = xb[i][0]
        ub[i] = xb[i][1]
    

    # optimizer summary
    print_summary(optimizer, n_dv, obj_scale, its, accu, x0, xb)

    opt_prob = pyoptsparse.Optimization(optimizer + ' pyOptSparse Optimization',obj_f)
    opt_prob.addVarGroup("x",n_dv,'c',lower=lb[0], upper=ub[0],value=x0[0])
    opt_prob.addObj('obj')
    opt_prob.addConGroup("con", len(project.config.OPT_CONSTRAINT['INEQUALITY']),lower=lb[0],upper=ub[0])


    if (optimizer == 'SLSQP'):
        opt  =SLSQP(options = optOptions)
        opt.setOption('IPRINT', 1)
        opt.setOption('ACC', accu)
        opt.setOption('MAXIT', its)

    # User-defined sensitivities
    sens = obj_df

    ############################################
    # Call the optimizer
    sol = opt(opt_prob, sens=sens, p1=project)
    print(sol)
    ############################################

    return 0

# -------FUNCTION DEFINITIONS   

def print_summary(optimizer_name, n_dv, obj_scale, its, accu, x0, xb):
  # optimizer summary
  sys.stdout.write(optimizer_name + ' parameters:\n')
  sys.stdout.write('Number of design variables: ' + str(n_dv) + '\n')
  sys.stdout.write('Objective function scaling factor: ' + str(obj_scale) + '\n')
  sys.stdout.write('Maximum number of iterations: ' + str(its) + '\n')
  sys.stdout.write('Requested accuracy: ' + str(accu) + '\n')
  sys.stdout.write('Initial guess for the independent variable(s): ' + str(x0) + '\n')
  sys.stdout.write('Lower and upper bound for each independent variable: ' + str(xb) + '\n\n')


# FUNCTION DEFINITION FOR OBJECTIVE FUNCTION CALCULATION
def obj_f(xdict, project):
    x = xdict["x"]
    funcs={}
    obj_list = project.obj_f(x)
    obj = 0
    for this_obj in obj_list:
        obj = obj+this_obj
    funcs["obj"] = obj
    
    #---------------------------------------#
    iqcon_list = project.con_cieq(x)
    iqcon = 0
    for this_iqcon in iqcon_list:
        iqcon = iqcon + this_iqcon
    funcs["con"] = iqcon_list


    fail = False

    return funcs, fail

# FUNCTION DEFINITION FOR OBJECTIVE SENSITIVITY CALCULATION
def obj_df(xdict, funcsDict, project):
    x = xdict["x"]
    funcsSens = {}
    funcsSens["obj"] = {}
    funcsSens["obj"]["x"] = {}
    dobj_list = project.obj_df(x)
    dobj=[0.0]*len(dobj_list[0])
    
    for this_dobj in dobj_list:
        idv=0
        for this_dv_dobj in this_dobj:
            dobj[idv] = dobj[idv]+this_dv_dobj
            idv+=1
    dobj = array( dobj )
    list1 = dobj.tolist()
    funcsSens["obj"]["x"] = list1
    #---------------------------------------#
    uncsSens = {}
    funcsSens["con"] = {}
    funcsSens["con"]["x"] = {}
    dcon_list = project.con_dcieq(x)
    dcon=[0.0]*len(dcon_list[0])
    
    for this_dcon in dcon_list:
        idv=0
        for this_dv_dcon in this_dcon:
            dcon[idv] = dcon[idv]+this_dv_dcon
            idv+=1
    dcon = array( dcon )
    list2 = dcon.tolist()
    funcsSens["con"]["x"] = dcon_list
    
    fail = False

    return funcsSens, fail

# FUNCTION DEFINITION FOR EQUALITY CONSTRAINT CALCULATION
def con_ceq(x,project):
    """ cons = con_ceq(x,project)
        
        Equality Constraint Functions
        SU2 Project interface to scipy.fmin_slsqp
        
        su2:         ceq(x) = 0.0, list[nceq]
        scipy_slsqp: ceq(x) = 0.0, ndarray[nceq]
    """
    
    cons = project.con_ceq(x)
    
    if cons: cons = array(cons)
    else:    cons = zeros([0])
        
    return cons

# FUNCTION DEFINITION FOR EQUALITY CONSTRAINT CALCULATION
def con_cieq(x,project):
    """ cons = con_cieq(x,project)
        
        Inequality Constraints
        SU2 Project interface to scipy.fmin_slsqp
        
        su2:         cieq(x) < 0.0, list[ncieq]
        scipy_slsqp: cieq(x) > 0.0, ndarray[ncieq]
    """
    
    cons = project.con_cieq(x)
    
    if cons: cons = array(cons)
    else:    cons = zeros([0])
    
    return cons

# FUNCTION DEFINITION FOR INEQUALITY CONSTRAINT SENSITIVITY CALCULATION
def con_dcieq(x,project):
    """ dcons = con_dcieq(x,project)
        
        Inequality Constraint Gradients
        SU2 Project interface to scipy.fmin_slsqp
        
        su2:         dcieq(x), list[ncieq x dim]
        scipy_slsqp: dcieq(x), ndarray[ncieq x dim]
    """
    
    dcons = project.con_dcieq(x)
    
    dim = project.n_dv
    if dcons: dcons = array(dcons)
    else:     dcons = zeros([0,dim])
    
    return dcons

# FUNCTION DEFINITION FOR EQUALITY CONSTRAINT SENSITIVITY CALCULATION
def con_dceq(x,project):
    """ dcons = con_dceq(x,project)
        
        Equality Constraint Gradients
        SU2 Project interface to scipy.fmin_slsqp
        
        su2:         dceq(x), list[nceq x dim]
        scipy_slsqp: dceq(x), ndarray[nceq x dim]
    """
    
    dcons = project.con_dceq(x)

    dim = project.n_dv
    if dcons: dcons = array(dcons)
    else:     dcons = zeros([0,dim])
    
    return dcons






