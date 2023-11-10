/*!
 * \file CDriverL.cpp
 * \brief The main sourbrouintes for performing singlezone computations
 * \author P. Ranjan
 * \version 7.2.0 "Columbia"

 * This is a part of forked SU2 library
 * Queries specific to SU2_MDO can be directed to 
 * Prateek Ranjan : Email -> pranjan2@illinois.edu

 * SU2_MDO is a lightweight build of SU2_CFD and 
 * performs calculation in a single zone only

 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2021, SU2 Contributors (cf. AUTHORS.md)
 *
 * SU2 is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * SU2 is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE. See the GNU
 * Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with SU2. If not, see <http://www.gnu.org/licenses/>.
 */

 #include "../../include/drivers/CDriverL.hpp"
 #include "../../include/definition_structure.hpp"
 #include "../../../Common/include/geometry/CPhysicalGeometry.hpp"


#include "../../include/solvers/CSolverFactory.hpp"

#include "../../include/output/COutputFactory.hpp"
#include "../../include/output/COutput.hpp"

#include "../../include/output/COutputLegacy.hpp"

#include "../../../Common/include/interface_interpolation/CInterpolator.hpp"
#include "../../../Common/include/interface_interpolation/CInterpolatorFactory.hpp"

#include "../../include/numerics/template.hpp"
#include "../../include/numerics/transition.hpp"

#include "../../include/numerics/heat.hpp"
#include "../../include/numerics/flow/convection/roe.hpp"
#include "../../include/numerics/flow/convection/fds.hpp"
#include "../../include/numerics/flow/convection/fvs.hpp"
#include "../../include/numerics/flow/flow_diffusion.hpp"
#include "../../include/numerics/flow/flow_sources.hpp"

#include "../../include/numerics/turbulent/turb_convection.hpp"
#include "../../include/numerics/turbulent/turb_diffusion.hpp"
#include "../../include/numerics/turbulent/turb_sources.hpp"

#include "../../include/integration/CIntegrationFactory.hpp"

#include "../../include/iteration/CIterationFactory.hpp"

#include "../../../Common/include/parallelization/omp_structure.hpp"


#include <cassert>

#ifdef VTUNEPROF
#include <ittnotify.h>
#endif
#include <fenv.h>


CDriver::CDriver(char* confFile, unsigned short val_nZone, SU2_Comm MPICommunicator, bool dummy_geo) :
  config_file_name(confFile), StartTime(0.0), StopTime(0.0), UsedTime(0.0),
  TimeIter(0), nZone(val_nZone), StopCalc(false), fsi(false), fem_solver(false), dry_run(dummy_geo) 
  
{

/*--- Initialize Medipack (must also be here so it is initialized from python) ---*/
#ifdef HAVE_MPI
  #if defined(CODI_REVERSE_TYPE) || defined(CODI_FORWARD_TYPE)
    SU2_MPI::Init_AMPI();
  #endif
#endif

SU2_MPI::SetComm(MPICommunicator);

  rank = SU2_MPI::GetRank();
  size = SU2_MPI::GetSize();


/*--- Start timer to track preprocessing for benchmarking. ---*/

  StartTime = SU2_MPI::Wtime();

  /*--- Initialize containers with null --- */

  SetContainers_Null();

  /*--- Preprocessing of the config files. ---*/

  Input_Preprocessing(config_container, driver_config);

  /*--- Retrieve dimension from mesh file ---*/

  nDim = CConfig::GetnDim(config_container[ZONE_0]->GetMesh_FileName(),
                          config_container[ZONE_0]->GetMesh_FileFormat());

/*--- Output preprocessing ---*/

  Output_Preprocessing(config_container, driver_config, output_container, driver_output);                          

for (iZone = 0; iZone < nZone; iZone++) 
{

    /*--- Read the number of instances for each zone ---*/

    nInst[iZone] = config_container[iZone]->GetnTimeInstances();

    geometry_container[iZone]    = new CGeometry**    [nInst[iZone]] ();
    iteration_container[iZone]   = new CIteration*    [nInst[iZone]] ();
    solver_container[iZone]      = new CSolver***     [nInst[iZone]] ();
    integration_container[iZone] = new CIntegration** [nInst[iZone]] ();
    numerics_container[iZone]    = new CNumerics****  [nInst[iZone]] ();
    grid_movement[iZone]         = new CVolumetricMovement* [nInst[iZone]] ();

    /*--- Allocate transfer and interpolation container --- */

    interface_container[iZone]    = new CInterface*[nZone] ();
    interpolator_container[iZone].resize(nZone);

    for (iInst = 0; iInst < nInst[iZone]; iInst++) {

      config_container[iZone]->SetiInst(iInst);

      /*--- Preprocessing of the geometry for all zones. In this routine, the edge-
       based data structure is constructed, i.e. node and cell neighbors are
       identified and linked, face areas and volumes of the dual mesh cells are
       computed, and the multigrid levels are created using an agglomeration procedure. ---*/

      Geometrical_Preprocessing(config_container[iZone], geometry_container[iZone][iInst], dry_run);

    }
  }

  /*--- Before we proceed with the zone loop we have to compute the wall distances.
     * This computation depends on all zones at once. ---*/
  if (rank == MASTER_NODE)
    cout << "Computing wall distances." << endl;

  CGeometry::ComputeWallDistance(config_container, geometry_container);

   if (rank == MASTER_NODE)
   {
    cout << "Begining the main solver loop" << endl;
   }

   for (iZone = 0; iZone < nZone; iZone++) 
   {

    for (iInst = 0; iInst < nInst[iZone]; iInst++)
    {

      /*--- Definition of the solver class: solver_container[#ZONES][#INSTANCES][#MG_GRIDS][#EQ_SYSTEMS].
       The solver classes are specific to a particular set of governing equations,
       and they contain the subroutines with instructions for computing each spatial
       term of the PDE, i.e. loops over the edges to compute convective and viscous
       fluxes, loops over the nodes to compute source terms, and routines for
       imposing various boundary condition type for the PDE. ---*/


    Solver_Preprocessing(config_container[iZone], geometry_container[iZone][iInst], solver_container[iZone][iInst]);

      /*--- Definition of the numerical method class:
       numerics_container[#ZONES][#INSTANCES][#MG_GRIDS][#EQ_SYSTEMS][#EQ_TERMS].
       The numerics class contains the implementation of the numerical methods for
       evaluating convective or viscous fluxes between any two nodes in the edge-based
       data structure (centered, upwind, galerkin), as well as any source terms
       (piecewise constant reconstruction) evaluated in each dual mesh volume. ---*/


    Numerics_Preprocessing(config_container[iZone], geometry_container[iZone][iInst],
                             solver_container[iZone][iInst], numerics_container[iZone][iInst]);

      /*--- Definition of the integration class: integration_container[#ZONES][#INSTANCES][#EQ_SYSTEMS].
       The integration class orchestrates the execution of the spatial integration
       subroutines contained in the solver class (including multigrid) for computing
       the residual at each node, R(U) and then integrates the equations to a
       steady state or time-accurately. ---*/

    Integration_Preprocessing(config_container[iZone], solver_container[iZone][iInst][MESH_0],
                                integration_container[iZone][iInst]);

      /*--- Instantiate the type of physics iteration to be executed within each zone. For
       example, one can execute the same physics across multiple zones (mixing plane),
       different physics in different zones (fluid-structure interaction), or couple multiple
       systems tightly within a single zone by creating a new iteration class (e.g., RANS). ---*/

      Iteration_Preprocessing(config_container[iZone], iteration_container[iZone][iInst]);

      /*--- Dynamic mesh processing.  ---*/

      DynamicMesh_Preprocessing(config_container[iZone], geometry_container[iZone][iInst], solver_container[iZone][iInst],
                                iteration_container[iZone][iInst], grid_movement[iZone][iInst], surface_movement[iZone]);
      /*--- Static mesh processing.  ---*/

      StaticMesh_Preprocessing(config_container[iZone], geometry_container[iZone][iInst]);

    } 
   }          

   PythonInterface_Preprocessing(config_container, geometry_container, solver_container);


  /*--- Preprocessing time is reported now, but not included in the next compute portion. ---*/
  /*--- Start Time is computed at Line 119. ---*/
  StopTime = SU2_MPI::Wtime();

/*--- Compute/print the total time for performance benchmarking. ---*/

  UsedTime = StopTime-StartTime;
  UsedTimePreproc    = UsedTime;
  /* -- Initialize compute and output times here. ---*/
  UsedTimeCompute    = 0.0;
  UsedTimeOutput     = 0.0;
  IterCount          = 0;
  OutputCount        = 0;
  MDOFs              = 0.0;
  MDOFsDomain        = 0.0;
  Mpoints            = 0.0;
  MpointsDomain      = 0.0;

  for (iZone = 0; iZone < nZone; iZone++) 
  {
    Mpoints       += geometry_container[iZone][INST_0][MESH_0]->GetGlobal_nPoint()/(1.0e6);
    MpointsDomain += geometry_container[iZone][INST_0][MESH_0]->GetGlobal_nPointDomain()/(1.0e6);
    MDOFs         += DOFsPerPoint*geometry_container[iZone][INST_0][MESH_0]->GetGlobal_nPoint()/(1.0e6);
    MDOFsDomain   += DOFsPerPoint*geometry_container[iZone][INST_0][MESH_0]->GetGlobal_nPointDomain()/(1.0e6);
  }

  /*--- Reset timer for compute/output performance benchmarking. ---*/

  StopTime = SU2_MPI::Wtime();

  /*--- Compute/print the total time for performance benchmarking. ---*/

  UsedTime = StopTime-StartTime;
  UsedTimePreproc = UsedTime;

  /*--- Reset timer for compute performance benchmarking. ---*/

  StartTime = SU2_MPI::Wtime();

}


void CDriver::SetContainers_Null(){

  /*--- Create pointers to all of the classes that may be used throughout
   the SU2_CFD code. In general, the pointers are instantiated down a
   hierarchy over all zones, multigrid levels, equation sets, and equation
   terms as described in the comments below. ---*/

  ConvHist_file                  = nullptr;
  iteration_container            = nullptr;
  output_container               = nullptr;
  integration_container          = nullptr;
  geometry_container             = nullptr;
  solver_container               = nullptr;
  numerics_container             = nullptr;
  config_container               = nullptr;
  surface_movement               = nullptr;
  grid_movement                  = nullptr;
  FFDBox                         = nullptr;
  interface_container            = nullptr;
  interface_types                = nullptr;
  nInst                          = nullptr;

/*--- Definition and of the containers for all possible zones. ---*/

  iteration_container            = new CIteration**[nZone] ();
  solver_container               = new CSolver****[nZone] ();
  integration_container          = new CIntegration***[nZone] ();
  numerics_container             = new CNumerics*****[nZone] ();
  config_container               = new CConfig*[nZone] ();
  geometry_container             = new CGeometry***[nZone] ();
  surface_movement               = new CSurfaceMovement*[nZone] ();
  grid_movement                  = new CVolumetricMovement**[nZone] ();
  FFDBox                         = new CFreeFormDefBox**[nZone] ();
  interpolator_container.resize(nZone);
  interface_container            = new CInterface**[nZone] ();
  interface_types                = new unsigned short*[nZone] ();
  output_container               = new COutput*[nZone] ();
  nInst                          = new unsigned short[nZone] ();
  driver_config                  = nullptr;
  driver_output                  = nullptr;

for (iZone = 0; iZone < nZone; iZone++) 
  {
    interface_types[iZone] = new unsigned short[nZone];
    nInst[iZone] = 1;
  }

  strcpy(runtime_file_name, "runtime.dat");

}

void CDriver::Postprocessing() {

  const bool wrt_perf = config_container[ZONE_0]->GetWrt_Performance();

    /*--- Output some information to the console. ---*/

  if (rank == MASTER_NODE) 
  
  {

    /*--- Print out the number of non-physical points and reconstructions ---*/

    if (config_container[ZONE_0]->GetNonphysical_Points() > 0)
      cout << "Warning: there are " << config_container[ZONE_0]->GetNonphysical_Points() << " non-physical points in the solution." << endl;
    if (config_container[ZONE_0]->GetNonphysical_Reconstr() > 0)
      cout << "Warning: " << config_container[ZONE_0]->GetNonphysical_Reconstr() << " reconstructed states for upwinding are non-physical." << endl;
  }

if (rank == MASTER_NODE)
    cout << endl <<"------------------------- Solver Postprocessing -------------------------" << endl;

  for (iZone = 0; iZone < nZone; iZone++) {
    for (iInst = 0; iInst < nInst[iZone]; iInst++){
      Numerics_Postprocessing(numerics_container[iZone], solver_container[iZone][iInst],
          geometry_container[iZone][iInst], config_container[iZone], iInst);
    }
    delete [] numerics_container[iZone];
  }
  delete [] numerics_container;
  if (rank == MASTER_NODE) cout << "Deleted CNumerics container." << endl;


for (iZone = 0; iZone < nZone; iZone++) 
{
    for (iInst = 0; iInst < nInst[iZone]; iInst++){

      Integration_Postprocessing(integration_container[iZone],
          geometry_container[iZone][iInst],
          config_container[iZone],
          iInst);
    }
        delete [] integration_container[iZone];
}

delete [] integration_container;


  if (rank == MASTER_NODE) cout << "Deleted CIntegration container." << endl;

  for (iZone = 0; iZone < nZone; iZone++) 
  {
    for (iInst = 0; iInst < nInst[iZone]; iInst++){
      Solver_Postprocessing(solver_container[iZone],
          geometry_container[iZone][iInst],
          config_container[iZone],
          iInst);
    }
    delete [] solver_container[iZone];
  }
  delete [] solver_container;
  if (rank == MASTER_NODE) cout << "Deleted CSolver container." << endl;

  for (iZone = 0; iZone < nZone; iZone++) 
  {
    for (iInst = 0; iInst < nInst[iZone]; iInst++)
      delete iteration_container[iZone][iInst];
    delete [] iteration_container[iZone];
  }
  delete [] iteration_container;
  if (rank == MASTER_NODE) cout << "Deleted CIteration container." << endl;

if (interface_container != nullptr) 
{
    for (iZone = 0; iZone < nZone; iZone++) 
    {
      if (interface_container[iZone] != nullptr) 
      {
        for (unsigned short jZone = 0; jZone < nZone; jZone++)
          delete interface_container[iZone][jZone];
        delete [] interface_container[iZone];
      }
    }
    delete [] interface_container;
    if (rank == MASTER_NODE) cout << "Deleted CInterface container." << endl;
  }

if (interface_types != nullptr) 
{
    for (iZone = 0; iZone < nZone; iZone++)
      delete [] interface_types[iZone];
    delete [] interface_types;
  }

for (iZone = 0; iZone < nZone; iZone++) 
{
    if (geometry_container[iZone] != nullptr) 
    {
      for (iInst = 0; iInst < nInst[iZone]; iInst++)
      {
        for (unsigned short iMGlevel = 0; iMGlevel < config_container[iZone]->GetnMGLevels()+1; iMGlevel++)
          delete geometry_container[iZone][iInst][iMGlevel];
        delete [] geometry_container[iZone][iInst];
      }
      delete [] geometry_container[iZone];
    }
  }
  delete [] geometry_container;
  if (rank == MASTER_NODE) cout << "Deleted CGeometry container." << endl;

  for (iZone = 0; iZone < nZone; iZone++) 
  {
    delete [] FFDBox[iZone];
  }

  delete [] FFDBox;
  if (rank == MASTER_NODE) cout << "Deleted CFreeFormDefBox class." << endl;

  for (iZone = 0; iZone < nZone; iZone++) 
  {
    delete surface_movement[iZone];
  }
  delete [] surface_movement;
  if (rank == MASTER_NODE) cout << "Deleted CSurfaceMovement class." << endl;

  for (iZone = 0; iZone < nZone; iZone++) 
  {
    for (iInst = 0; iInst < nInst[iZone]; iInst++)
      delete grid_movement[iZone][iInst];
    delete [] grid_movement[iZone];
  }
  delete [] grid_movement;
  if (rank == MASTER_NODE) cout << "Deleted CVolumetricMovement class." << endl;

/*--- Output profiling information ---*/
  // Note that for now this is called only by a single thread, but all
  // necessary variables have been made thread private for safety (tick/tock)!!

  config_container[ZONE_0]->SetProfilingCSV();
  config_container[ZONE_0]->GEMMProfilingCSV();

  /*--- Deallocate config container ---*/
  if (config_container!= nullptr) {
    for (iZone = 0; iZone < nZone; iZone++)
      delete config_container[iZone];
    delete [] config_container;
  }
  delete driver_config;
  if (rank == MASTER_NODE) cout << "Deleted CConfig container." << endl;

  delete [] nInst;
  if (rank == MASTER_NODE) cout << "Deleted nInst container." << endl;

/*--- Deallocate output container ---*/

  if (output_container!= nullptr) 
  {
    for (iZone = 0; iZone < nZone; iZone++)
      delete output_container[iZone];
    delete [] output_container;
  }  

delete driver_output;

  if (rank == MASTER_NODE) cout << "Deleted COutput class." << endl;

  if (rank == MASTER_NODE) cout << "-------------------------------------------------------------------------" << endl;

  /*--- Stop the timer and output the final performance summary. ---*/

  StopTime = SU2_MPI::Wtime();

  UsedTime = StopTime-StartTime;
  UsedTimeCompute += UsedTime;

  if ((rank == MASTER_NODE) && (wrt_perf)) 
  {
    su2double TotalTime = UsedTimePreproc + UsedTimeCompute + UsedTimeOutput;
    
    cout << endl << endl <<"-------------------------- Performance Summary --------------------------" << endl;
    cout << " SU2 Naive Profiling Results:" << endl;
    cout << setw(25) << "Wall-clock time (hrs):" << setw(12) << (TotalTime)/(60.0*60.0) << " | ";
    cout << setw(20) << "Core-hrs:" << setw(12) << size*TotalTime/(60.0*60.0) << endl;
    cout << setw(25) << "Cores:" << setw(12) << size << " | ";
    cout << setw(20) << "DOFs/point:" << setw(12) << DOFsPerPoint << endl;
    cout << setw(25) << "Points/core:" << setw(12) << 1.0e6*MpointsDomain/size << " | ";
    cout << setw(20) << "Ghost points/core:" << setw(12) << 1.0e6*(Mpoints-MpointsDomain)/size << endl;
    cout << setw(25) << "Ghost/Owned Point Ratio:" << setw(12) << (Mpoints-MpointsDomain)/MpointsDomain << " | " << endl;
    cout << endl;
    cout << "Preprocessing phase:" << endl;
    cout << setw(25) << "Preproc. Time (s):"  << setw(12)<< UsedTimePreproc << " | ";
    cout << setw(20) << "Preproc. Time (%):" << setw(12)<< ((UsedTimePreproc * 100.0) / (TotalTime)) << endl;
    cout << endl;
    cout << "Compute phase:" << endl;
    cout << setw(25) << "Compute Time (s):"  << setw(12)<< UsedTimeCompute << " | ";
    cout << setw(20) << "Compute Time (%):" << setw(12)<< ((UsedTimeCompute * 100.0) / (TotalTime)) << endl;
    cout << setw(25) << "Iteration count:"  << setw(12)<< IterCount << " | ";
    if (IterCount != 0) {
      cout << setw(20) << "Avg. s/iter:" << setw(12)<< UsedTimeCompute/IterCount << endl;
      cout << setw(25) << "Core-s/iter/Mpoints:" << setw(12)<< size*UsedTimeCompute/IterCount/Mpoints << " | ";
      cout << setw(20) << "Mpoints/s:" << setw(12)<< Mpoints*IterCount/UsedTimeCompute << endl;
    } else cout << endl;
    cout << endl;
    cout << "Output phase:" << endl;
    cout << setw(25) << "Output Time (s):"  << setw(12)<< UsedTimeOutput << " | ";
    cout << setw(20) << "Output Time (%):" << setw(12)<< ((UsedTimeOutput * 100.0) / (TotalTime)) << endl;
    cout << setw(25) << "Output count:" << setw(12)<< OutputCount << " | ";
    if (OutputCount != 0) {
      cout << setw(20)<< "Avg. s/output:" << setw(12)<< UsedTimeOutput/OutputCount << endl;
      if (BandwidthSum > 0) {
        cout << setw(25)<< "Restart Aggr. BW (MB/s):" << setw(12)<< BandwidthSum/OutputCount << " | ";
        cout << setw(20)<< "MB/s/core:" << setw(12)<< BandwidthSum/OutputCount/size << endl;
      }
    } else cout << endl;
    cout << "-------------------------------------------------------------------------" << endl;
    cout << endl;
  }    

   /*--- Exit the solver cleanly ---*/

  if (rank == MASTER_NODE)
    cout << endl <<"------------------------- Exit Success (SU2_CFD) ------------------------" << endl << endl;

}

void CDriver::Input_Preprocessing(CConfig **&config, CConfig *&driver_config) 
{

  char zone_file_name[MAX_STRING_SIZE];

  /*--- Initialize the configuration of the driver ---*/

  driver_config = new CConfig(config_file_name, SU2_COMPONENT::SU2_MDO, false);

  for (iZone = 0; iZone < nZone; iZone++) {

    if (rank == MASTER_NODE){
      cout  << endl << "Parsing config file for zone " << iZone << endl;
    }
    /*--- Definition of the configuration option class for all zones. In this
     constructor, the input configuration file is parsed and all options are
     read and stored. ---*/

    if (driver_config->GetnConfigFiles() > 0){

      strcpy(zone_file_name, driver_config->GetConfigFilename(iZone).c_str());
      config[iZone] = new CConfig(driver_config, zone_file_name, SU2_COMPONENT::SU2_MDO, iZone, nZone, true);
    }
    else{
      config[iZone] = new CConfig(driver_config, config_file_name, SU2_COMPONENT::SU2_MDO, iZone, nZone, true);
    }

    /*--- Set the MPI communicator ---*/

    config[iZone]->SetMPICommunicator(SU2_MPI::GetComm());
  }


}

void CDriver::Geometrical_Preprocessing(CConfig* config, CGeometry **&geometry, bool dummy)
{
   if (!dummy)
   {
    if (rank == MASTER_NODE)
      cout << endl <<"------------------- Geometry Preprocessing ( Zone " << config->GetiZone() <<" ) -------------------" << endl;

      Geometrical_Preprocessing_FVM(config, geometry);

   }

   /*--- Computation of positive surface area in the z-plane which is used for
     the calculation of force coefficient (non-dimensionalization). ---*/

  geometry[MESH_0]->SetPositive_ZArea(config);


  /*--- If activated by the compile directive, perform a partition analysis. ---*/
#if PARTITION
  if (!dummy)
  {
    if( fem_solver ) Partition_Analysis_FEM(geometry[MESH_0], config);
    else Partition_Analysis(geometry[MESH_0], config);
  }
#endif























