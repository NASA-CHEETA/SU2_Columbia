/*!
 * \file CMDODriver.cpp
 * \brief The main subroutines for driving coupled aero-elastic problems.
 * \author P. Ranjan
 * \version 7.2.0 "Columbia"
 *
 * SU2 Project Website: https://su2code.github.io
 *
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

#include "../../include/drivers/CMDODriver.hpp"
#include "../../include/definition_structure.hpp"
#include "../../include/output/COutput.hpp"
#include "../../include/iteration/CIteration.hpp"

#include "../../include/output/CMeshOutput.hpp"

#include "../../include/precice.hpp"

#include <string.h>

CMDODriver::CMDODriver(char* confFile,
                       unsigned short val_nZone,
                       SU2_Comm MPICommunicator) : CDriver(confFile,
                                                          val_nZone,
                                                          MPICommunicator,
                                                          false) {

  /*--- Initialize the counter for TimeIter ---*/
  TimeIter = 0;
}

CMDODriver::~CMDODriver(void) 
{

}

void CMDODriver::StartSolver() 
{

  StartTime = SU2_MPI::Wtime();

  config_container[ZONE_0]->Set_StartTime(StartTime);

  /*---Output counter---*/
  unsigned long counter = 0;

  /*--- Main external loop of the solver. Runs for the number of time steps required. ---*/

  if (rank == MASTER_NODE)
  {
    cout << endl <<"------------------------------ Begin Forward Analysis -----------------------------" << endl;  
  }
    /*---See if Unsteady MDA/MDO object needs to be created ---*/
    enable_Steady_MDO = config_container[ZONE_0]->GetSMDO_Mode();
    enable_Unsteady_MDO = config_container[ZONE_0]->GetUMDO_Mode();

    /*--- Set the initial time iteration to the restart iteration. ---*/
    if (config_container[ZONE_0]->GetRestart() && driver_config->GetTime_Domain())
    {
      TimeIter = config_container[ZONE_0]->GetRestart_Iter();
    }
  
    /*---If Steady MDA is required, create a coupling object ----*/
    if (enable_Steady_MDO) 
    {
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
    }

    /*---If Unsteady MDA is required, create a coupling object ----*/
    if (enable_Unsteady_MDO) 
    {
      precice = new Precice(config_container[ZONE_0]->GetpreCICE_ConfigFileName(),rank, size,config_container, geometry_container, solver_container, grid_movement);
      dt = new double(config_container[ZONE_0]->GetDelta_UnstTimeND());

      if (rank == MASTER_NODE)
      {
        std::cout << "------------------------------ Initialize Coupling Interface --------------------------------" << std::endl;
      }

      max_precice_dt = new double(precice->initialize());

      if (rank == MASTER_NODE)
      {
        std::cout << "------------------------------- Interface Initialization Complete ---------------------------------" << std::endl;
      }
    }  

    /*---Get the time at which aero-elastic state must be computed---*/
    target_time = config_container[ZONE_0]->GetTargTimeIter();


  if (rank == MASTER_NODE)
  {
    cout << endl <<"Simulation Run using the MDO Driver" << endl;
   
    if (driver_config->GetTime_Domain())
    {
      cout << "The simulation will run for "
           << driver_config->GetnTime_Iter() - config_container[ZONE_0]->GetRestart_Iter() << " time steps or until implicit convergence" << endl;
    }
  }



  if (enable_Steady_MDO)
  {
    if (rank == MASTER_NODE)
    {
      std::cout <<"Aeroelasitc simulations will begin at: " << target_time << std::endl;
    }
  
  
    while ((TimeIter < config_container[ZONE_0]->GetnTime_Iter()) &&!enable_Steady_MDO || (TimeIter < config_container[ZONE_0]->GetnTime_Iter()) && enable_Steady_MDO && precice->isCouplingOngoing() ||(TimeIter < config_container[ZONE_0]->GetnTime_Iter()) && enable_Steady_MDO)
    {

      
        if (TimeIter == target_time)
      {
      /*---Save the current fluid state---*///
        precice->saveOldState(&StopCalc, dt);
      }
      
      /*---- Deform the mesh here based on surface displacements of previous advance---*/
      Preprocess(TimeIter);

              

      /*---Run implicit iteration---*/
      RunMDO(TimeIter);  
    
      /*--- Compute tractions baed on current fluid state---*/
      Postprocess();

      /*--- Update the solution for dual time stepping strategy ---*/
      Update();
    
      /*--- Monitor the computations after each iteration. ---*/
      Monitor(TimeIter);

      /*---Output the required fields to files---*/
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
            std::cout << "Write deformed mesh to file" <<std::endl;
          }
          output_container[ZONE_0]->WriteToFile(config_container[ZONE_0],geometry_container[ZONE_0][INST_0][MESH_0], MESH, config_container[ZONE_0]->GetMesh_Out_FileName());


          break;
        }  


     /* 
    if (enable_Steady_MDO && !(precice->isCouplingOngoing()))
      {
        if (rank==MASTER_NODE)
        {
          std::cout <<"Static aero-elastic solution converged"<<std::endl;
        }

        if(rank==MASTER_NODE)
        {
          std::cout<<"Writing fluid field at aero-elastic equillibrium"<<std::endl;
        }
        Output(TimeIter);

        if (rank == MASTER_NODE)
        {
          std::cout << "Loading mesh data to master node" << std::endl;
        }
        output_container[ZONE_0]->Load_Data(geometry_container[ZONE_0][INST_0][MESH_0], config_container[ZONE_0], solver_container[ZONE_0][INST_0][MESH_0]);

        if (rank == MASTER_NODE)
        {
          std::cout << "Write deformed mesh to file" <<std::endl;
        }
        output_container[ZONE_0]->WriteToFile(config_container[ZONE_0],geometry_container[ZONE_0][INST_0][MESH_0], MESH, config_container[ZONE_0]->GetMesh_Out_FileName());

        break;
      }

      */


      if ((TimeIter == target_time) && (precice->isCouplingOngoing()))
      {
        /*---Compute surface tractions and recieve displaced surface---*/
        *max_precice_dt = precice->advance(*dt);
        
        /* Only update the grid after the mesh as been deformed by CCX---*/
    
       /* 
        auto iteration = iteration_container[ZONE_0][INST_0];

        iteration->SetGrid_Movement(geometry_container[ZONE_0][INST_0],surface_movement[ZONE_0],
                                grid_movement[ZONE_0][INST_0], solver_container[ZONE_0][INST_0],
                                config_container[ZONE_0], 0, 100);
        */
       
        
        /*---Stay at the current time---*/
        TimeIter--;      

        /*---Reload the fluid state---*/
        precice->reloadOldState(&StopCalc, dt);

      } 

      

 



      /*--- If the convergence criteria has been met, terminate the simulation. ---*/
      //if (StopCalc) break;
      counter++;
      TimeIter++;

    }
  } // Steady-state MDA loop ends

  /*-----------------------------------------------------------------------------------------------------------------------*/

  if (enable_Unsteady_MDO)
  {  
    while ((TimeIter < config_container[ZONE_0]->GetnTime_Iter()) &&!enable_Unsteady_MDO || (TimeIter < config_container[ZONE_0]->GetnTime_Iter()) && enable_Unsteady_MDO && precice->isCouplingOngoing() ||(TimeIter < config_container[ZONE_0]->GetnTime_Iter()) && enable_Unsteady_MDO)
    {

      /*---Save old state for implicit coupling---*/
      if (precice->isActionRequired(precice->getCowic()))
      {
        precice->saveOldState(&StopCalc, dt);
      }
      
      /*---Adjust the time step for sub-cycling---*/
      dt = min(max_precice_dt,dt);
      config_container[ZONE_0]->SetDelta_UnstTimeND(*dt);

      /*--- Perform some preprocessing before starting the time-step simulation. ---*/
      Preprocess(TimeIter);

      Run();  
    
      /*--- Perform some postprocessing on the solution before the update ---*/
      Postprocess();

      /*--- Update the solution for dual time stepping strategy ---*/
      Update();
    
      /*--- Monitor the computations after each iteration. ---*/
      Monitor(TimeIter);

      /*---Advance the aero-elastic state---*/
      *max_precice_dt = precice->advance(*dt);


    
     
      bool suppress_output = false;
    
      /* if(precice_usage && precice->isActionRequired(precice->getCoric())) */
      if(enable_Unsteady_MDO && precice->isActionRequired(precice->getCoric()))
      {
      //Stay at the same iteration number if preCICE is not converged and reload to the state before the current iteration
      TimeIter--;
      precice->reloadOldState(&StopCalc, dt);
      suppress_output = true;
      }


      /*--- Save iteration solution for libROM ---*/
      if (config_container[MESH_0]->GetSave_libROM()) 
      {
        solver_container[ZONE_0][INST_0][MESH_0][FLOW_SOL]->SavelibROM(geometry_container[ZONE_0][INST_0][MESH_0],
                                                                     config_container[ZONE_0], StopCalc);
      }


      Implicit_Output(TimeIter, suppress_output);

      /*--- If the convergence criteria has been met, terminate the simulation. ---*/

      if (StopCalc) break;

      TimeIter++;
    }
  } // Unsteady-state MDA loop ends

  if (enable_Steady_MDO || enable_Unsteady_MDO)
  {
    if (precice != NULL)
    {
      if (rank == MASTER_NODE)
      {
        std::cout <<"---------------------------------------------------------"<<std::endl;
        std::cout << "-------------------Deleted MDO object-------------------"<<std::endl;
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

}


void CMDODriver::Preprocess(unsigned long TimeIter) {

  /*--- Set runtime option ---*/

  Runtime_Options();

  /*--- Set the current time iteration in the config ---*/

  config_container[ZONE_0]->SetTimeIter(TimeIter);

  /*--- Store the current physical time in the config container, as
   this can be used for verification / MMS. This should also be more
   general once the drivers are more stable. ---*/

  if (config_container[ZONE_0]->GetTime_Marching() != TIME_MARCHING::STEADY)
    //if (enable_Unsteady_MDO)
    {
      config_container[ZONE_0]->SetPhysicalTime(static_cast<su2double>(TimeIter)*config_container[ZONE_0]->GetDelta_UnstTimeND());
    }
    //else if (enable_Steady_MDO)
    else
    {
      config_container[ZONE_0]->SetPhysicalTime(0.0);
    }  

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

  /*---Perform a dynamic mesh update only at MDO target time */
  if ((enable_Steady_MDO) && (TimeIter == target_time) || (enable_Unsteady_MDO))
  {
    DynamicMeshUpdate(TimeIter);
  }
}

void CMDODriver::PreprocessMDO(unsigned long TimeIter, unsigned long counter) {

  /*--- Set runtime option ---*/

  Runtime_Options();

  /*--- Set the current time iteration in the config ---*/

  config_container[ZONE_0]->SetTimeIter(TimeIter);

  /*--- Store the current physical time in the config container, as
   this can be used for verification / MMS. This should also be more
   general once the drivers are more stable. ---*/

  if (config_container[ZONE_0]->GetTime_Marching() != TIME_MARCHING::STEADY)
    //if (enable_Unsteady_MDO)
    {
      config_container[ZONE_0]->SetPhysicalTime(static_cast<su2double>(TimeIter)*config_container[ZONE_0]->GetDelta_UnstTimeND());
    }
    //else if (enable_Steady_MDO)
    else
    {
      config_container[ZONE_0]->SetPhysicalTime(0.0);
    }  

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

  /*---Perform a dynamic mesh update only at MDO target time */
  if ((enable_Steady_MDO) && (TimeIter == target_time) || (enable_Unsteady_MDO))
  {
    if (rank == MASTER_NODE)
    {
      std::cout<<"Performing dynamic mesh update for aero-elastic iteration: "<< counter << std::endl;
    }
    DynamicMeshUpdate(counter);
  }
}

void CMDODriver::Run() 
{

  unsigned long OuterIter = 0;
  config_container[ZONE_0]->SetOuterIter(OuterIter);


    /*--- Iterate the zone as a block, either to convergence or to a max number of iterations ---*/
    iteration_container[ZONE_0][INST_0]->Solve(output_container[ZONE_0], integration_container, geometry_container, solver_container,
        numerics_container, config_container, surface_movement, grid_movement, FFDBox, ZONE_0, INST_0);  

}

void CMDODriver::RunMDO(unsigned long TimeIter) 
{

  unsigned long OuterIter = 0;
  config_container[ZONE_0]->SetOuterIter(OuterIter);

    /*--- Iterate the zone as a block, either to convergence or to a max number of inner iterations for Implicit MDA ---*/
    iteration_container[ZONE_0][INST_0]->MDOSolve(output_container[ZONE_0], integration_container, geometry_container, solver_container,
        numerics_container, config_container, surface_movement, grid_movement, FFDBox, ZONE_0, INST_0, TimeIter);
        
}

void CMDODriver::Postprocess() {

  iteration_container[ZONE_0][INST_0]->Postprocess(output_container[ZONE_0], integration_container, geometry_container, solver_container,
      numerics_container, config_container, surface_movement, grid_movement, FFDBox, ZONE_0, INST_0);

  /*--- A corrector step can help preventing numerical instabilities ---*/

  if (config_container[ZONE_0]->GetRelaxation())
    iteration_container[ZONE_0][INST_0]->Relaxation(output_container[ZONE_0], integration_container, geometry_container, solver_container,
        numerics_container, config_container, surface_movement, grid_movement, FFDBox, ZONE_0, INST_0);

}

void CMDODriver::Update() {

  iteration_container[ZONE_0][INST_0]->Update(output_container[ZONE_0], integration_container, geometry_container,
        solver_container, numerics_container, config_container,
        surface_movement, grid_movement, FFDBox, ZONE_0, INST_0);

}

void CMDODriver::Implicit_Output(unsigned long TimeIter, bool suppress_output)
{
  if (!suppress_output)
  {
    std::cout << "Writing solution for converged aero-elastic solution " << std::endl;
    /*--- Time the output for performance benchmarking. ---*/

    StopTime = SU2_MPI::Wtime();

    UsedTimeCompute += StopTime-StartTime;

    StartTime = SU2_MPI::Wtime();

    bool wrote_files = output_container[ZONE_0]->SetResult_Files(geometry_container[ZONE_0][INST_0][MESH_0],
                                                               config_container[ZONE_0],
                                                               solver_container[ZONE_0][INST_0][MESH_0],
                                                               TimeIter, StopCalc);
    if (wrote_files)
    {
      StopTime = SU2_MPI::Wtime();
      UsedTimeOutput += StopTime-StartTime;
      OutputCount++;
      BandwidthSum = config_container[ZONE_0]->GetRestart_Bandwidth_Agg();
      StartTime = SU2_MPI::Wtime();
      config_container[ZONE_0]->Set_StartTime(StartTime);
    }
  }
} 

void CMDODriver::Output(unsigned long TimeIter) 
{
  
  /*--- Time the output for performance benchmarking. ---*/

  StopTime = SU2_MPI::Wtime();

  UsedTimeCompute += StopTime-StartTime;

  StartTime = SU2_MPI::Wtime();

  bool wrote_files = output_container[ZONE_0]->SetResult_Files(geometry_container[ZONE_0][INST_0][MESH_0],
                                                               config_container[ZONE_0],
                                                               solver_container[ZONE_0][INST_0][MESH_0],
                                                               TimeIter, StopCalc);
  if (wrote_files){

    StopTime = SU2_MPI::Wtime();

    UsedTimeOutput += StopTime-StartTime;
    OutputCount++;
    BandwidthSum = config_container[ZONE_0]->GetRestart_Bandwidth_Agg();

    StartTime = SU2_MPI::Wtime();

    //config_container[ZONE_0]->Set_StartTime(StartTime);
  }
}

void CMDODriver::DynamicMeshUpdate(unsigned long TimeIter) {

  auto iteration = iteration_container[ZONE_0][INST_0];

  /*--- Legacy dynamic mesh update - Only if GRID_MOVEMENT = YES ---*/
  
  if (config_container[ZONE_0]->GetGrid_Movement()) 
  {
    if (rank == MASTER_NODE)
    {
      std::cout <<"Performing grid deformation using LEGACY class"<<std::endl;
    }
    iteration->SetGrid_Movement(geometry_container[ZONE_0][INST_0],surface_movement[ZONE_0],
                                grid_movement[ZONE_0][INST_0], solver_container[ZONE_0][INST_0],
                                config_container[ZONE_0], 0, 0);
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

bool CMDODriver::Monitor(unsigned long TimeIter)
{

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

  if (TimeDomain == NO)
  {

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

  if ((TimeDomain == YES) && (TimeIter == target_time))
  {
    InnerConvergence     = output_container[ZONE_0]->GetConvergence();
    MaxIterationsReached = InnerIter+1 >= nInnerIter;

    if ((MaxIterationsReached || InnerConvergence) && (rank == MASTER_NODE)) 
    {
      cout << endl << "----------------------------- MDA Solver Exit -------------------------------" << endl;
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

void CMDODriver::Runtime_Options(){

  ifstream runtime_configfile;

  /*--- Try to open the runtime config file ---*/

  runtime_configfile.open(runtime_file_name, ios::in);

  /*--- If succeeded create a temporary config object ---*/

  if (runtime_configfile.good()){
    CConfig *runtime = new CConfig(runtime_file_name, config_container[ZONE_0]);
    delete runtime;
  }

}

bool CMDODriver::GetTimeConvergence() const{
  return output_container[ZONE_0]->GetCauchyCorrectedTimeConvergence(config_container[ZONE_0]);
}
