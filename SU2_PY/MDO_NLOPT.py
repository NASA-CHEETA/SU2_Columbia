#!/usr/bin/env python

## \file MDO.py
#  \brief Python script for performing MDO.
#  \author Prateek Ranjan
#  \version 7.2.0 "Blackbird"
import nlopt
import os, sys, shutil
from optparse import OptionParser
sys.path.append(os.environ['SU2_RUN'])
import SU2


# -------------------------------------------------------------------
#  Main
# -------------------------------------------------------------------

def main():

    parser=OptionParser()
    parser.add_option("-f", "--file", dest="filename",
                      help="read config from FILE", metavar="FILE")
    parser.add_option("-r", "--name", dest="projectname", default='',
                      help="try to restart from project file NAME", metavar="NAME")
    parser.add_option("-n", "--partitions", dest="partitions", default=1,
                      help="number of PARTITIONS", metavar="PARTITIONS")
    parser.add_option("-g", "--gradient", dest="gradient", default="DISCRETE_ADJOINT",
                      help="Method for computing the GRADIENT (CONTINUOUS_ADJOINT, DISCRETE_ADJOINT, FINDIFF, NONE)", metavar="GRADIENT")
    parser.add_option("-o", "--optimization", dest="optimization", default="SLSQP",
                      help="OPTIMIZATION techique (SLSQP, CG, BFGS, POWELL)", metavar="OPTIMIZATION")
    parser.add_option("-q", "--quiet", dest="quiet", default="True",
                      help="True/False Quiet all SU2 output (optimizer output only)", metavar="QUIET")
    parser.add_option("-z", "--zones", dest="nzones", default="1",
                      help="Number of Zones", metavar="ZONES")


    (options, args)=parser.parse_args()

    # process inputs
    options.partitions  = int( options.partitions )
    options.quiet       = options.quiet.upper() == 'TRUE'
    options.gradient    = options.gradient.upper()
    options.nzones      = int( options.nzones )

    shape_optimization( options.filename    ,
                    options.projectname ,
                    options.partitions  ,
                    options.gradient    ,
                    options.optimization ,
                    options.quiet       ,
                    options.nzones      )


def shape_optimization(filename                           ,
                        projectname = ''                   ,
                        partitions  = 0                    ,
                        gradient    = 'DISCRETE_ADJOINT' ,
                        optimization = 'SLSQP'             ,
                        quiet       = False                ,
                        nzones      = 1                    ):

#Setup Configuration for Optimizer
    config = SU2.io.Config(filename)
    config.NUMBER_PART = partitions
    config.NZONES      = int( nzones )
    if quiet: config.CONSOLE = 'CONCISE'
    config.GRADIENT_METHOD = gradient

    its               = int ( config.OPT_ITERATIONS )                      # number of opt iterations
    bound_upper       = float ( config.OPT_BOUND_UPPER )                   # variable bound to be scaled by the line search
    bound_lower       = float ( config.OPT_BOUND_LOWER )                   # variable bound to be scaled by the line search
    relax_factor      = float ( config.OPT_RELAX_FACTOR )                  # line search scale
    gradient_factor   = float ( config.OPT_GRADIENT_FACTOR )               # objective function and gradient scale
    def_dv            = config.DEFINITION_DV                               # complete definition of the desing variable
    n_dv              = sum(def_dv['SIZE'])                                # number of design variables
    accu              = float ( config.OPT_ACCURACY ) * gradient_factor    # optimizer accuracy
    x0                = [0.0]*n_dv # initial design
    xb_low            = [float(bound_lower)/float(relax_factor)]*n_dv      # lower dv bound it includes the line search acceleration factor
    xb_up             = [float(bound_upper)/float(relax_factor)]*n_dv      # upper dv bound it includes the line search acceleration fa
    xb                = list(zip(xb_low, xb_up)) # design bounds                                

    # State
    state = SU2.io.State()
    state.find_files(config)

    # add restart files to state.FILES
    if config.get('TIME_DOMAIN', 'NO') == 'YES' and config.get('RESTART_SOL', 'NO') == 'YES' and gradient != 'CONTINUOUS_ADJOINT':
        restart_name = config['RESTART_FILENAME'].split('.')[0]
        restart_filename = restart_name + '_' + str(int(config['RESTART_ITER'])-1).zfill(5) + '.dat'
        if not os.path.isfile(restart_filename): # throw, if restart files does not exist
            sys.exit("Error: Restart file <" + restart_filename + "> not found.")
        state['FILES']['RESTART_FILE_1'] = restart_filename

    # Project

    print("Setting up SU2 project class ....")

    if os.path.exists(projectname):
        project = SU2.io.load_data(projectname)
        project.config = config
    else:
        project = SU2.opt.Project(config,state)

    #Optimize

    #SU2.opt.nlopt_optim(project)
    SU2.opt.nlopt_optim(project,x0,xb,10,accu)

    #Rename project file
    if projectname:
        shutil.move('project.pkl',projectname)

    return project

# -------------------------------------------------------------------
#  Run Main Program
# -------------------------------------------------------------------
#print(" Pre-prosessing steps for MDO Complete !")
# this is only accessed if running from command prompt
if __name__ == '__main__':
    main()           

    


