/*!
 * \file driver CStatic_MDO.cpp
 * \brief The main subroutines for driving MDA/O problems.
 * \author P. Ranjan
 * \version 2.0.0 "Columbia"
 *
 * SU2 Project Website: https://su2code.github.io
 *
 * The SU2 Project is maintained by the SU2 Foundation
 * (http://su2foundation.org)
 *
 * Copyright 2012-2021, SU2 Contributors (cf. AUTHORS.md)
 *
 * SU2 is free software; 
 * can redistribute it and/or
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

#include "../../include/drivers/CStaticMDODriver.hpp"
#include "../../include/definition_structure.hpp"
#include "../../include/output/COutput.hpp"
#include "../../include/iteration/CIteration.hpp"

#include "../../include/adapter/precice.hpp"

CStaticMDODriver::CStaticMDODriver(char* confFile,
                       unsigned short val_nZone,
                       SU2_Comm MPICommunicator) : CDriver(confFile,
                                                          val_nZone,
                                                          MPICommunicator,
                                                          false)
                                                          {

  /*--- Initialize the counter for TimeIter ---*/
  TimeIter = 0;
}

CStaticMDODriver::~CStaticMDODriver(void) 
{

}

void CStaticMDODriver::StartSolver()
{

  StartTime = SU2_MPI::Wtime();

  config_container[ZONE_0]->Set_StartTime(StartTime);

  /*---See if Steady MDA/O object needs to be created ---*/
  bool enable_Steady_MDO = config_container[ZONE_0]->GetSMDO_Mode();



  if (rank == MASTER_NODE)
  {
    cout << endl <<"------------------------------ Begin Forward Analysis -----------------------------" << endl;
  }




  if (rank == MASTER_NODE)
  {
    cout << endl <<"Simulation Run using the STATIC aero-elasticity Driver" << endl;
  }
  
  /* Create the coupling object */
 
  precice = new Precice(config_container[ZONE_0]->GetpreCICE_ConfigFileName(),rank, size,config_container, geometry_container, solver_container, grid_movement);
  dt = new double(1);

  if (rank == MASTER_NODE)
  {
    std::cout << "------------------------------ Initialize Coupling Interface --------------------------------" << std::endl;
  }

  max_precice_dt = new double(precice->initialize());

  if (rank == MASTER_NODE)
  {
    std::cout << "------------------------------- Interface Initialization Complete ---------------------------------" << std::endl;
  }

  /*---Get the time at which aero-elastic state must be computed---*/
  //target_time = config_container[ZONE_0]->GetTargTimeIter();
  target_time = 0;
  /*---Initialize counter to toggle enble/disable CL_Driver---*/
  int counter = 0;

  /*--- Main external loop of the solver. Runs for the number of time steps required. ---*/


  /*--- Set the initial time iteration to the restart iteration. ---*/
  if (config_container[ZONE_0]->GetRestart() && driver_config->GetTime_Domain())
    TimeIter = config_container[ZONE_0]->GetRestart_Iter();

   while ((TimeIter < config_container[ZONE_0]->GetnTime_Iter()) &&!enable_Steady_MDO || (TimeIter < config_container[ZONE_0]->GetnTime_Iter()) && enable_Steady_MDO && precice->isCouplingOngoing() ||(TimeIter < config_container[ZONE_0]->GetnTime_Iter()) && enable_Steady_MDO)
  /*--- Run the problem until the number of time iterations required is reached. ---*/
  {

    if (TimeIter == target_time)
    {
     /*---Save the current fluid state---*///
     precice->saveOldStaticState(&StopCalc, dt);    
    }

    /* Deform the mesh here based on surface displacements of previous advance */
    Preprocess(TimeIter);

              
    /* Run implicit iteration */
    RunSMDO(counter);  
    
    /* Compute tractions baed on current fluid state */
    Postprocess();

    /* Update the solution for dual time stepping strategy */
    Update();
    
    /* Monitor the computations after each iteration */
    Monitor(TimeIter);


    /* Output the required fields to files */
    if (!(precice->isCouplingOngoing()))
    {
      if(rank==MASTER_NODE)
      {
        std::cout<<"Aero-elastic solution converged!"<<std::endl;
        std::cout<<"Writing fluid field at aero-elastic equillibrium"<<std::endl;
      }

      Output(TimeIter);

      /*---Output the deformed mesh---*/
      if (rank == MASTER_NODE)
      {
        std::cout << "Loading mesh data to master node" << std::endl;
      }
      output_container[ZONE_0]->Load_Data(geometry_container[ZONE_0][INST_0][MESH_0], config_container[ZONE_0], solver_container[ZONE_0][INST_0][MESH_0]);

      if (rank == MASTER_NODE)
      {
        std::cout << "Writing deformed mesh to file" <<std::endl;
      }
      output_container[ZONE_0]->WriteToFile(config_container[ZONE_0],geometry_container[ZONE_0][INST_0][MESH_0], OUTPUT_TYPE::MESH, config_container[ZONE_0]->GetMesh_Out_FileName());
      break;
    }

    if ((TimeIter == target_time) && (precice->isCouplingOngoing()))
    {
      /* Compute surface tractions and recieve displaced surface */
      *max_precice_dt = precice->advance(*dt);

      /* Disable the CL_Driver for remaining implicit iterations */
      //config_container[ZONE_0]->Set_CL_Driver_Mode(false);
               
      /* Stay at the current time */
      TimeIter--;      

      /* Reload the fluid state */
      precice->reloadOldStaticState(&StopCalc, dt);
    }

    /*--- If the convergence criteria has been met, terminate the simulation. ---*/
   
    counter++;
    TimeIter++;

  } /*---Implicit loop ends here---*/

    if (precice != NULL)
    {
      if (rank == MASTER_NODE)
      {
        std::cout <<"---------------------------------------------------------"<<std::endl;
        std::cout <<"-------------------Deleted MDA/O object--------------------"<<std::endl;
        std::cout <<"---------------------------------------------------------"<<std::endl;
      }
      delete precice;
    }

    if (dt != NULL)
    {
      delete dt;
    }

    if (max_precice_dt != NULL)
    {
      delete max_precice_dt;
    }
}



void CStaticMDODriver::Preprocess(unsigned long TimeIter) {

  /*--- Set runtime option ---*/

  Runtime_Options();

  /*--- Set the current time iteration in the config ---*/

  config_container[ZONE_0]->SetTimeIter(TimeIter);

  /*--- Store the current physical time in the config container, as
   this can be used for verification / MMS. This should also be more
   general once the drivers are more stable. ---*/

  if (config_container[ZONE_0]->GetTime_Marching() != TIME_MARCHING::STEADY)
    config_container[ZONE_0]->SetPhysicalTime(static_cast<su2double>(TimeIter)*config_container[ZONE_0]->GetDelta_UnstTimeND());
  else
    config_container[ZONE_0]->SetPhysicalTime(0.0);

  /*--- Set the initial condition for EULER/N-S/RANS ---------------------------------------------*/
  if (config_container[ZONE_0]->GetFluidProblem()) {
    solver_container[ZONE_0][INST_0][MESH_0][FLOW_SOL]->SetInitialCondition(geometry_container[ZONE_0][INST_0],
                                                                            solver_container[ZONE_0][INST_0],
                                                                            config_container[ZONE_0], TimeIter);
  }
  else if (config_container[ZONE_0]->GetHeatProblem()) {
    /*--- Set the initial condition for HEAT equation ---------------------------------------------*/
    solver_container[ZONE_0][INST_0][MESH_0][HEAT_SOL]->SetInitialCondition(geometry_container[ZONE_0][INST_0],
                                                                            solver_container[ZONE_0][INST_0],
                                                                            config_container[ZONE_0], TimeIter);
  }

  SU2_MPI::Barrier(SU2_MPI::GetComm());

  /*--- Run a predictor step ---*/
  if (config_container[ZONE_0]->GetPredictor())
    iteration_container[ZONE_0][INST_0]->Predictor(output_container[ZONE_0], integration_container, geometry_container, solver_container,
        numerics_container, config_container, surface_movement, grid_movement, FFDBox, ZONE_0, INST_0);

  /*--- Perform a dynamic mesh update if required. ---*/
  /*--- For the Disc.Adj. of a case with (rigidly) moving grid, the appropriate
          mesh cordinates are read from the restart files. ---*/
 // if (!(config_container[ZONE_0]->GetGrid_Movement() && config_container[ZONE_0]->GetDiscrete_Adjoint()))
    DynamicMeshUpdate(TimeIter);

}

void CStaticMDODriver::RunSMDO(int counter)
{

  unsigned long OuterIter = 0;
  config_container[ZONE_0]->SetOuterIter(OuterIter);

    /*--- Iterate the zone as a block, either to convergence or to a max number of inner iterations for Implicit MDA ---*/
    iteration_container[ZONE_0][INST_0]->MDOSolve(output_container[ZONE_0], integration_container, geometry_container, solver_container,
        numerics_container, config_container, surface_movement, grid_movement, FFDBox, ZONE_0, INST_0, counter);

}

void CStaticMDODriver::Run() 
{

  unsigned long OuterIter = 0;
  config_container[ZONE_0]->SetOuterIter(OuterIter);

  /*--- Iterate the zone as a block, either to convergence or to a max number of iterations ---*/
  iteration_container[ZONE_0][INST_0]->Solve(output_container[ZONE_0], integration_container, geometry_container, solver_container,
        numerics_container, config_container, surface_movement, grid_movement, FFDBox, ZONE_0, INST_0);

}



void CStaticMDODriver::Postprocess() {

  iteration_container[ZONE_0][INST_0]->Postprocess(output_container[ZONE_0], integration_container, geometry_container, solver_container,
      numerics_container, config_container, surface_movement, grid_movement, FFDBox, ZONE_0, INST_0);

  /*--- A corrector step can help preventing numerical instabilities ---*/

  if (config_container[ZONE_0]->GetRelaxation())
    iteration_container[ZONE_0][INST_0]->Relaxation(output_container[ZONE_0], integration_container, geometry_container, solver_container,
        numerics_container, config_container, surface_movement, grid_movement, FFDBox, ZONE_0, INST_0);

}

void CStaticMDODriver::Update() {

  iteration_container[ZONE_0][INST_0]->Update(output_container[ZONE_0], integration_container, geometry_container,
        solver_container, numerics_container, config_container,
        surface_movement, grid_movement, FFDBox, ZONE_0, INST_0);

}

void CStaticMDODriver::Output(unsigned long TimeIter) {

  /*--- Time the output for performance benchmarking. ---*/

  StopTime = SU2_MPI::Wtime();

  UsedTimeCompute += StopTime-StartTime;

  StartTime = SU2_MPI::Wtime();

  bool wrote_files = output_container[ZONE_0]->SetResult_Files(geometry_container[ZONE_0][INST_0][MESH_0],
                                                               config_container[ZONE_0],
                                                               solver_container[ZONE_0][INST_0][MESH_0],
                                                               0, 1);




  if (wrote_files)
  {

    if (rank == MASTER_NODE)
    {
      std::cout << " Output files written! " << std::endl;
    }
    StopTime = SU2_MPI::Wtime();

    UsedTimeOutput += StopTime-StartTime;
    OutputCount++;
    BandwidthSum = config_container[ZONE_0]->GetRestart_Bandwidth_Agg();

    StartTime = SU2_MPI::Wtime();

    config_container[ZONE_0]->Set_StartTime(StartTime);
  }
}

void CStaticMDODriver::DynamicMeshUpdate(unsigned long TimeIter) {

  auto iteration = iteration_container[ZONE_0][INST_0];

  /*--- Legacy dynamic mesh update - Only if GRID_MOVEMENT = YES ---*/
  if (config_container[ZONE_0]->GetGrid_Movement()) {
    iteration->SetGrid_Movement(geometry_container[ZONE_0][INST_0],surface_movement[ZONE_0],
                                grid_movement[ZONE_0][INST_0], solver_container[ZONE_0][INST_0],
                                config_container[ZONE_0], 0, TimeIter);
  }

  /*--- New solver - all the other routines in SetGrid_Movement should be adapted to this one ---*/
  /*--- Works if DEFORM_MESH = YES ---*/
  iteration->SetMesh_Deformation(geometry_container[ZONE_0][INST_0],
                                 solver_container[ZONE_0][INST_0][MESH_0],
                                 numerics_container[ZONE_0][INST_0][MESH_0],
                                 config_container[ZONE_0], RECORDING::CLEAR_INDICES);

  /*--- Update the wall distances if the mesh was deformed. ---*/
  if (config_container[ZONE_0]->GetGrid_Movement() ||
      config_container[ZONE_0]->GetDeform_Mesh()) {
    CGeometry::ComputeWallDistance(config_container, geometry_container);
  }
}

bool CStaticMDODriver::Monitor(unsigned long TimeIter){

  unsigned long nInnerIter, InnerIter, nTimeIter;
  su2double MaxTime, CurTime;
  bool TimeDomain, InnerConvergence, TimeConvergence, FinalTimeReached, MaxIterationsReached;

  nInnerIter = config_container[ZONE_0]->GetnInner_Iter();
  InnerIter  = config_container[ZONE_0]->GetInnerIter();
  nTimeIter  = config_container[ZONE_0]->GetnTime_Iter();
  MaxTime    = config_container[ZONE_0]->GetMax_Time();
  CurTime    = output_container[ZONE_0]->GetHistoryFieldValue("CUR_TIME");

  TimeDomain = config_container[ZONE_0]->GetTime_Domain();


  /*--- Check whether the inner solver has converged --- */

  if (TimeDomain == NO){

    InnerConvergence     = output_container[ZONE_0]->GetConvergence();
    MaxIterationsReached = InnerIter+1 >= nInnerIter;

    if ((MaxIterationsReached || InnerConvergence) && (rank == MASTER_NODE)) {
      cout << endl << "----------------------------- Solver Exit -------------------------------" << endl;
      if (InnerConvergence) cout << "All convergence criteria satisfied." << endl;
      else cout << endl << "Maximum number of iterations reached (ITER = " << nInnerIter << ") before convergence." << endl;
      output_container[ZONE_0]->PrintConvergenceSummary();
      cout << "-------------------------------------------------------------------------" << endl;
    }

    StopCalc = MaxIterationsReached || InnerConvergence;
  }



  if (TimeDomain == YES) {

    /*--- Check whether the outer time integration has reached the final time ---*/

    TimeConvergence = GetTimeConvergence();

    FinalTimeReached     = CurTime >= MaxTime;
    MaxIterationsReached = TimeIter+1 >= nTimeIter;

    if ((FinalTimeReached || MaxIterationsReached || TimeConvergence) && (rank == MASTER_NODE)){
      cout << endl << "----------------------------- Solver Exit -------------------------------";
      if (TimeConvergence)     cout << endl << "All windowed time-averaged convergence criteria are fullfilled." << endl;
      if (FinalTimeReached)     cout << endl << "Maximum time reached (MAX_TIME = " << MaxTime << "s)." << endl;
      if (MaxIterationsReached) cout << endl << "Maximum number of time iterations reached (TIME_ITER = " << nTimeIter << ")." << endl;
      cout << "-------------------------------------------------------------------------" << endl;
    }
    StopCalc = FinalTimeReached || MaxIterationsReached|| TimeConvergence;
  }

  /*--- Reset the inner convergence --- */

  output_container[ZONE_0]->SetConvergence(false);

  /*--- Increase the total iteration count --- */

  IterCount += config_container[ZONE_0]->GetInnerIter()+1;

  return StopCalc;
}

void CStaticMDODriver::Runtime_Options(){

  ifstream runtime_configfile;

  /*--- Try to open the runtime config file ---*/

  runtime_configfile.open(runtime_file_name, ios::in);

  /*--- If succeeded create a temporary config object ---*/

  if (runtime_configfile.good()){
    CConfig *runtime = new CConfig(runtime_file_name, config_container[ZONE_0]);
    delete runtime;
  }

}

bool CStaticMDODriver::GetTimeConvergence() const{
  return output_container[ZONE_0]->GetCauchyCorrectedTimeConvergence(config_container[ZONE_0]);
}
