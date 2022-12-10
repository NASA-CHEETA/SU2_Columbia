/*!
* \file precice.cpp
* \brief Adapter class for coupling SU2 with preCICE for FSI.
* \author Prateek Ranjan ( adapted from Rush )
*/
#include <iostream>
#include <string.h>
#include <stdio.h>


#include "../include//adapter/precice.hpp"


/*---Main class description -----*/

  Precice::Precice(const std::string& preciceConfigurationFileName, int solverProcessIndex, int solverProcessSize, CConfig** config_container, CGeometry**** geometry_container, CSolver***** solver_container, CVolumetricMovement*** grid_movement)
  :
  coric(precice::constants::actionReadIterationCheckpoint()),
  cowic(precice::constants::actionWriteIterationCheckpoint()),
  solverProcessIndex(solverProcessIndex),
  solverProcessSize(solverProcessSize),
  solverInterface("SU2_CFD", preciceConfigurationFileName, solverProcessIndex, solverProcessSize),
  config_container(config_container),
  geometry_container(geometry_container),
  solver_container(solver_container),
  grid_movement(grid_movement)
  {
    /* Get dimension of the problem */
    nDim = geometry_container[ZONE_0][INST_0][MESH_0]->GetnDim();
  

    /* Initialize the coupling datasets to NULL */
    
    vertexIDs = NULL;
    forceID = NULL;
    displDeltaID = NULL;
    forces = NULL;
    displacementDeltas = NULL;

    /* Initialize the variables for implicit coupling */

    processWorkingOnWetSurface = true;
    verbosityLevel_high = config_container[ZONE_0]->GetpreCICE_VerbosityLevel_High();
    globalNumberWetSurfaces = config_container[ZONE_0]->GetpreCICE_NumberWetSurfaces();
    localNumberWetSurfaces = 0;
    valueMarkerWet = NULL;
    vertexSize = NULL;
    indexMarkerWetMappingLocalToGlobal = NULL;



    /* Get the total number of nodes in the domain */
    nPoint = geometry_container[ZONE_0][INST_0][MESH_0]->GetnPoint();

   

    /* Get the total number of variables in the domain */
    nVar = solver_container[ZONE_0][INST_0][MESH_0][FLOW_SOL]->GetnVar();

       /*---Get total number of markers---*/
    nMarkers_Global = config_container[ZONE_0]->GetnMarker_CfgFile();

    nMarkers_Local = config_container[ZONE_0]->GetnMarker_All();

    /*---Identify the marker for the FSI Interface ---*/
    FSI_NAME = config_container[ZONE_0]->GetpreCICE_WetSurfaceMarkerName();

  /*--- Get the marker ID for FSI surface---*/
    FSI_ID_Global = config_container[ZONE_0]->GetMarker_CfgFile_TagBound(FSI_NAME+to_string(0));
    FSI_ID_Local = config_container[ZONE_0]->GetMarker_All_TagBound(FSI_NAME+to_string(0));

    /* Initialize and allocate memory for arrays required for implicit coupling */
    Coord_Saved            = NULL;
    Coord_n_Saved          = NULL;
    Coord_n1_Saved         = NULL;
    Coord_p1_Saved         = NULL;
    GridVel_Saved          = NULL;
    GridVel_Grad_Saved     = NULL;
    dt_savedState          = 0;
    StopCalc_savedState    = false;
    solution_Saved         = NULL;
    solution_time_n_Saved  = NULL;
    solution_time_n1_Saved = NULL;

    Coord_Saved = new double*[nPoint];
    Coord_n_Saved = new double*[nPoint];
    Coord_n1_Saved = new double*[nPoint];
    Coord_p1_Saved = new double*[nPoint];
    GridVel_Saved = new double*[nPoint];
    GridVel_Grad_Saved = new double**[nPoint];
    solution_Saved = new double*[nPoint];
    solution_time_n_Saved = new double*[nPoint];
    solution_time_n1_Saved = new double*[nPoint];

    for (int iPoint = 0; iPoint < nPoint; iPoint++) 
    {
      Coord_Saved[iPoint] = new double[nDim];
      Coord_n_Saved[iPoint] = new double[nDim];
      Coord_n1_Saved[iPoint] = new double[nDim];
      Coord_p1_Saved[iPoint] = new double[nDim];
      GridVel_Saved[iPoint] = new double[nDim];
      GridVel_Grad_Saved[iPoint] = new double*[nDim];
      for (int iDim = 0; iDim < nDim; iDim++) 
      {
        GridVel_Grad_Saved[iPoint][iDim] = new double[nDim];
      }
      solution_Saved[iPoint] = new double[nVar];
      solution_time_n_Saved[iPoint] = new double[nVar];
      solution_time_n1_Saved[iPoint] = new double[nVar];
    }
  }


/*---Class destructor -----*/  
Precice::~Precice(void)
{
    for (int i = 0; i < localNumberWetSurfaces; i++) 
    {
      if (vertexIDs[i] != NULL)
        {
          delete [] vertexIDs[i];
        }
    }

    if (vertexIDs != NULL) 
    {
      delete [] vertexIDs;
    }
    if (forceID != NULL) 
    {
      delete [] forceID;
    }
    if (displDeltaID != NULL) 
    {
      delete [] displDeltaID;
    }
    if (valueMarkerWet != NULL) 
    {
      delete [] valueMarkerWet;
    }
    if (vertexSize != NULL) 
    {
      delete [] vertexSize;
    }
    if (indexMarkerWetMappingLocalToGlobal != NULL) 
    {
      delete [] indexMarkerWetMappingLocalToGlobal;
    }

    for (int iPoint = 0; iPoint < nPoint; iPoint++) 
    {
      if (Coord_Saved[iPoint] != NULL) 
      {
        delete [] Coord_Saved[iPoint];
      }
      if (Coord_n_Saved[iPoint] != NULL) 
      {
        delete [] Coord_n_Saved[iPoint];
      }
      if (Coord_n1_Saved[iPoint] != NULL) 
      {
        delete [] Coord_n1_Saved[iPoint];
      }
      if (Coord_p1_Saved[iPoint] != NULL) 
      {
        delete [] Coord_p1_Saved[iPoint];
      }
      if (GridVel_Saved[iPoint] != NULL) 
      {
        delete [] GridVel_Saved[iPoint];
      }
      for (int iDim = 0; iDim < nDim; iDim++) 
      {
        if (GridVel_Grad_Saved[iPoint][iDim] != NULL) 
        {
          delete [] GridVel_Grad_Saved[iPoint][iDim];
        }
      }
      if (GridVel_Grad_Saved[iPoint] != NULL) 
      {
        delete [] GridVel_Grad_Saved[iPoint];
      }
      if (solution_Saved[iPoint] != NULL) 
      {
        delete [] solution_Saved[iPoint];
      }
      if (solution_time_n_Saved[iPoint] != NULL) 
      {
        delete [] solution_time_n_Saved[iPoint];
      }
      if (solution_time_n1_Saved[iPoint] != NULL) 
      {
        delete [] solution_time_n1_Saved[iPoint];
      }
    }
    if (Coord_Saved != NULL) 
    {
      delete [] Coord_Saved;
    }
    if (Coord_n_Saved != NULL) 
    {
      delete [] Coord_n_Saved;
    }
    if (Coord_n1_Saved != NULL)
    {
      delete [] Coord_n1_Saved;
    }
    if (Coord_p1_Saved != NULL) 
    {
      delete [] Coord_p1_Saved;
    }
    if (GridVel_Saved != NULL) 
    {
      delete [] GridVel_Saved;
    }
    if (GridVel_Grad_Saved != NULL) 
    {
      delete [] GridVel_Grad_Saved;
    }
    if (solution_Saved != NULL) 
    {
      delete [] solution_Saved;
    }
    if (solution_time_n_Saved != NULL) 
    {
      delete [] solution_time_n_Saved;
    }
    if (solution_time_n1_Saved != NULL) 
    {
      delete [] solution_time_n1_Saved;
    }
  }


void Precice::check()
{
    std::cout << " Precice being called " << std::endl;
}


double Precice::initialize()
{
  bool Debug = false;
  /*---Check for dimensional consistency between SU2 and .xml file---*/
  if (solverInterface.getDimensions() != nDim)
  {
    std::cout << "Fluid Domain dimension mismatch " <<std::endl;
    std::cout << "Dimension in SU2: " << nDim << std::endl;
    std::cout << "Dimension in MDA config file: " << solverInterface.getDimensions() << std::endl;
    exit(EXIT_FAILURE);
  }

  if (solverProcessIndex == 0)
  {
    std::cout << "Dimension check complete " << std::endl;
  }


  
  
  /*---Check for total number of aero-elastic interfaces---*/
  
  if ( globalNumberWetSurfaces < 1)
  {
    std::cout << "Invalid number of aero-elastic interfaces." << std::endl;
    std::cout << "Number of aero-elastic surfaces: " << globalNumberWetSurfaces << std::endl;
    exit(EXIT_FAILURE);
  }


  else
  {

    /*---The next three are scalars ( single interface ) or vector pointers (multiple interfaces)---*/
    meshID = new int[globalNumberWetSurfaces];
    forceID = new int[globalNumberWetSurfaces];
    displDeltaID = new int[globalNumberWetSurfaces];

    /*---Get meshIDs from preCICE---*/
    for ( int i = 0; i < globalNumberWetSurfaces; i++)
    {
      meshID[i] = solverInterface.getMeshID("SU2_Mesh" + to_string(i));
    }
  }

  if ( solverProcessIndex == 0)
  {
    std::cout << "Mesh iD allocation complete " << std::endl;
  }
   
  /*---Determine the number of wet surfaces, that this process is working on, then loop over this number for all respective preCICE-related tasks---*/
  for (int i = 0; i < globalNumberWetSurfaces; i++) 
  {       
    if (config_container[ZONE_0]->GetMarker_All_TagBound(config_container[ZONE_0]->GetpreCICE_WetSurfaceMarkerName() + to_string(i)) == -1) 
    {
      //std::cout << "Process #" << solverProcessIndex << "/" << solverProcessSize-1 << ": Does not work on " << config_container[ZONE_0]->GetpreCICE_WetSurfaceMarkerName() << i << endl;
    } 
    else 
    {
      localNumberWetSurfaces++;
    }
  }
  
  if (Debug)
  { 
    std::cout << " Global Number of wet surfaces : " << globalNumberWetSurfaces << std::endl;
    std::cout << " Local number of wet surfaces  : " << localNumberWetSurfaces << std::endl;
    std::cout << " Dimensions of Fluid problem   : " << nDim << std::endl;
    std::cout << " Dimensions from XML File      : " << solverInterface.getDimensions() << std::endl;
    std::cout << " MeshID for each wetsurface    : " << meshID[0] << std::endl;
    std::cout << " Marker ID for FSI surface     : " << config_container[ZONE_0]->GetMarker_All_TagBound(config_container[ZONE_0]->GetpreCICE_WetSurfaceMarkerName() + to_string(0)) << std::endl;
  }

  if (solverProcessIndex == 0)
  {
    std::cout << "Finished determining number of wet surfaces " << std::endl;
  }

    unsigned long FSI_nVert = geometry_container[ZONE_0][INST_0][MESH_0]->nVertex[FSI_ID_Local];


  if (localNumberWetSurfaces < 1) 
  {
    processWorkingOnWetSurface = false;
  }



  if (processWorkingOnWetSurface) 
  {
    if ( solverProcessIndex == 0)
    {
      std::cout << "Discretizing Aero-elastic Interface! \n";
    }
    /*---Store the wet surface marker values in an array, which has the size equal to the number of wet surfaces actually being worked on by this process---*/
    valueMarkerWet = new short[localNumberWetSurfaces];
    indexMarkerWetMappingLocalToGlobal = new short[localNumberWetSurfaces];
    int j = 0;
    for (int i = 0; i < globalNumberWetSurfaces; i++) 
    {
      if (config_container[ZONE_0]->GetMarker_All_TagBound(config_container[ZONE_0]->GetpreCICE_WetSurfaceMarkerName() + to_string(i)) != -1) 
      {
        valueMarkerWet[j] = config_container[ZONE_0]->GetMarker_All_TagBound(config_container[ZONE_0]->GetpreCICE_WetSurfaceMarkerName() + to_string(i));
        indexMarkerWetMappingLocalToGlobal[j] = i;
        j++;
      }
    }
    vertexIDs = new int*[localNumberWetSurfaces];
  }
 
   if (solverProcessIndex == 0)
  { 
    std::cout << "Finished setting indices " << std::endl;
  } 


  if (processWorkingOnWetSurface) 
  {
    vertexSize = new unsigned long[localNumberWetSurfaces];

    for (int i = 0; i < localNumberWetSurfaces; i++) 
    {
      vertexSize[i] = geometry_container[ZONE_0][INST_0][MESH_0]->nVertex[valueMarkerWet[i]];

      /*--- coordinates of all nodes at the wet surface ---*/
      double coupleNodeCoord[vertexSize[i]][nDim]; 

      /*--- variable for storing the node indices - one at the time ---*/
      unsigned long iNode;  
      int l  =0;

      /*---Loop over the vertices of the (each) boundary---*/
      for (int iVertex = 0; iVertex < vertexSize[i]; iVertex++) 
      {
        /*---Get node number (= index) to vertex (= node) of the aeroelastic interface---*/
        iNode = geometry_container[ZONE_0][INST_0][MESH_0]->vertex[valueMarkerWet[i]][iVertex]->GetNode();
      //  std::cout << "ValueMakerWet: " << valueMarkerWet[i] << std::endl;
   
       /*---vector = geometry_container[ZONE_0][INST_0][MESH_0]->nodes->GetCoord(iNode);---*/
        if (geometry_container[ZONE_0][INST_0][MESH_0]->nodes->GetDomain(iNode))
        {
          /*---Unpdate the counter here---*/
          l++;
          /*---Get coordinates for nodes at aero-elastic interface --*/
          for (int iDim = 0; iDim < nDim; iDim++) 
          {  
          coupleNodeCoord[iVertex][iDim] = geometry_container[ZONE_0][INST_0][MESH_0]->nodes->GetCoord(iNode, iDim);
          }
        } 
        /* -- Print detailed surface information --*/
        
        
       //   std::cout << " Vertex: " << iVertex << std::setw(6) << " Node_index: " << iNode << std::setw(6) << " x: " << coupleNodeCoord[iVertex][0] << std::setw(6) << " y: " << coupleNodeCoord[iVertex][1] << std::setw(6) << " z: " << coupleNodeCoord[iVertex][2] << std::endl; 
        
      }


      if (solverProcessIndex == 0)
      {
        std::cout << "Finished acquiring node IDs and coordinates" << std::endl;
      }

      /*---------preCICE Internal Calculations -----*/
      
      /*---preCICE conform the coordinates of vertices (= points = nodes) at wet surface---*/
      double coords[vertexSize[i]*nDim];
      for (int iVertex = 0; iVertex < vertexSize[i]; iVertex++) 
      {
        for (int iDim = 0; iDim < nDim; iDim++) 
        {
          coords[iVertex*nDim + iDim] = coupleNodeCoord[iVertex][iDim];
        }
      }

      if (Debug)
      {
        std::cout << "Index Marker Local to Global : " << std::setw(10) << indexMarkerWetMappingLocalToGlobal[i] <<std::endl;
      }

      //preCICE internal
      vertexIDs[i] = new int[vertexSize[i]];

      solverInterface.setMeshVertices(meshID[indexMarkerWetMappingLocalToGlobal[i]], vertexSize[i], coords, vertexIDs[i]);


      forceID[indexMarkerWetMappingLocalToGlobal[i]] = solverInterface.getDataID("Forces" + to_string(indexMarkerWetMappingLocalToGlobal[i]), meshID[indexMarkerWetMappingLocalToGlobal[i]]);


      displDeltaID[indexMarkerWetMappingLocalToGlobal[i]] = solverInterface.getDataID("DisplacementDeltas" + to_string(indexMarkerWetMappingLocalToGlobal[i]), meshID[indexMarkerWetMappingLocalToGlobal[i]]);

      if (Debug)
      {
        std::cout << " ForceID    : " << forceID[0] << std::endl;
        std::cout << " DispDelID  : " << displDeltaID[0] << std::endl;
      }
    }

    for (int i = 0; i < globalNumberWetSurfaces; i++) 
    {
      bool flag = false;
      for (int j = 0; j < localNumberWetSurfaces; j++) 
      {
        if (indexMarkerWetMappingLocalToGlobal[j] == i) 
        {
          flag = true;
        }
      }
      if (!flag) 
      {
        solverInterface.setMeshVertices(meshID[i], 0, NULL, NULL);
        forceID[i] = solverInterface.getDataID("Forces" + to_string(i), meshID[i]);
        displDeltaID[i] = solverInterface.getDataID("DisplacementDeltas" + to_string(i), meshID[i]);
      }
    }
  }

  /*---Obtain Timestep across the interface---*/

  double precice_dt;

  if (solverProcessIndex == 0)
  {
    std::cout << "Establishing network link with CalculiX" << std::endl;
  }

  precice_dt = solverInterface.initialize();

  if (solverProcessIndex == 0)
  {
    std::cout << "Network link with CalculiX established" << std::endl;
  }
  
  return precice_dt;

}

double Precice::advance( double computedTimestepLength )
{
  bool Debug = false;
  if ( processWorkingOnWetSurface)
  {
    int procid = solverProcessIndex;
    unsigned long iPoint;
    double *iNormal;
    unsigned short iMarker;
    unsigned long iVertex;
    unsigned short iDim;

          
    /*---Number of vertices on FSI surface---*/
    unsigned long FSI_nVert = geometry_container[ZONE_0][INST_0][MESH_0]->nVertex[FSI_ID_Local];

   if (Debug)
   { 
      std::cout << "Fluid Grid Points : " << std::setw(6) << nPoint << "# Vertices on surface : " << std::setw(6) << FSI_nVert << std::endl;
   }
    /*---Two-dimensional array consisting of all tractions---*/
    double FSI_Trac[FSI_nVert][nDim];

    /*---Loop over all Markers to get the tractions at vertices---*/  

      for (int iVertex = 0; iVertex < FSI_nVert; iVertex++)
      {
            for (iDim = 0; iDim < nDim; iDim++)
            {
              FSI_Trac[iVertex][iDim] = solver_container[ZONE_0][INST_0][MESH_0][FLOW_SOL]->GetVertexTractions(FSI_ID_Local,iVertex,iDim);
            }
            if (Debug)
            {
              std::cout << "MarkerID : " << std::setw(6) << FSI_ID_Local << std::setw(6) << " VertexID: " << std::setw(6) << iVertex << std::setw(6) << " Tx: " << std::setw(6) << FSI_Trac[iVertex][0] << std::setw(6) << " Ty: " << std::setw(6) << FSI_Trac[iVertex][1] << std::setw(6) << " Tz: " << std::setw(6) << FSI_Trac[iVertex][2] <<std::endl;
            }  
      }

    

    /*---Convert FSI_Trac[][] to row-major form ---*/

    forces = new double[FSI_nVert*nDim];

    /*---Begin loop over all vertices at the FSI surface ---*/
    for (int iVertex = 0; iVertex < FSI_nVert; iVertex++)
    {                        
      for (iDim = 0; iDim < nDim; iDim++)
      {
        /*---Check if the color of the node matches the MPI rank of this proces.
             Only write forces if the node originally belongs to this process---*/
        iPoint = geometry_container[ZONE_0][INST_0][MESH_0]->vertex[FSI_ID_Local][iVertex]->GetNode();

        if (geometry_container[ZONE_0][INST_0][MESH_0]->nodes->GetColor(iPoint) == solverProcessIndex)
        {
          forces[iVertex*nDim + iDim] =  FSI_Trac[iVertex][iDim];
        }
       
        else
        {
          forces[iVertex*nDim + iDim] = 0;
        }

        
      }
    }

    /*---Write force across the TCP I/O interface ---*/

    solverInterface.writeBlockVectorData(forceID[indexMarkerWetMappingLocalToGlobal[0]], vertexSize[0], vertexIDs[0], forces);

    if (procid == 0)
    {
      std::cout << "Aerodynamic tractions transmitted to elastic domain " << std::endl;
    }

    /*---De-allocate force pointer---*/

    if (forces != NULL)
    {
      delete [] forces;
    }

    /*---Advace solverInterface---*/
    double max_precice_dt;
    max_precice_dt = solverInterface.advance( computedTimestepLength );

    /*---Read displacement deltas from elastic domain---*/

    // Recieve displacement deltas in row-major form of array 

    double displacementDeltas_su2[vertexSize[0]][nDim]; 
    displacementDeltas = new double[vertexSize[0]*nDim];

    
    solverInterface.readBlockVectorData(displDeltaID[indexMarkerWetMappingLocalToGlobal[0]], vertexSize[0], vertexIDs[0], displacementDeltas);

    /* Re-arrage the elastic inputs ---*/

   // std::cout << "FSI nVert: " << FSI_nVert << std::endl;

   // std::cout << "Reading vertex size: " << vertexSize[0] <<std::endl;

    for (int iVertex = 0; iVertex < FSI_nVert; iVertex++) 
    {
      for (int iDim = 0; iDim < nDim; iDim++) 
      {
        displacementDeltas_su2[iVertex][iDim] = displacementDeltas[iVertex*nDim + iDim];
        //std::cout << "Dispx " << setw(12) << displacementDeltas[iVertex*nDim + 0 ]<< " DispY " << setw(12) << displacementDeltas[iVertex*nDim + 1 ] << " DispZ " << setw(12) << displacementDeltas[iVertex*nDim + 2 ] << std::endl;
        //printf(" DispX %lf, DispY %lf, DispZ %lf \n", displacementDeltas[iVertex*nDim + 0], displacementDeltas[iVertex*nDim + 1 ], displacementDeltas[iVertex*nDim + 2 ]);
      }
    }
    /*---De-allocate memeory---*/

    if (displacementDeltas != NULL) 
    {
      delete [] displacementDeltas;
    }

    /*---Update the surface coordinates ---*/

    /*---Begin looping over all vertices at the interface---*/

    for (int iVertex = 0; iVertex < FSI_nVert; iVertex++)
    {
      geometry_container[ZONE_0][INST_0][MESH_0]->vertex[FSI_ID_Local][iVertex]->SetVarCoord(displacementDeltas_su2[iVertex]);
     // geometry_container[ZONE_0][INST_0][MESH_0]->vertex[FSI_ID_Local][iVertex]->AddVarCoord(displacementDeltas_su2[iVertex]);  
    }
    return max_precice_dt;
  }
  
  /* If the process does not have the AERO-elastic interface */
  else
  {
    /* Do not compute the forces. Just advance the solverInterface */
    double max_precice_dt;
    max_precice_dt = solverInterface.advance( computedTimestepLength );
    return max_precice_dt;
  } 
   /*---Advance ends here ---*/
}


//void Precice::saveOldState( bool *StopCalc, double *dt )
void Precice ::saveOldState( bool *StopCalc, double *dt )
{
  if (solverProcessIndex == 0)
  {
    std::cout << "Saving current state for implicit calculations" << std::endl;
  }

  /*---Begin loop over ALL grid points in the fluid domain---*/
  
  

  for (int iPoint = 0; iPoint < nPoint; iPoint++) 
  {
    for (int iVar = 0; iVar < nVar; iVar++) 
    {
      //Save solutions at last and current time step
      solution_Saved[iPoint][iVar] = (solver_container[ZONE_0][INST_0][MESH_0][FLOW_SOL]->GetNodes()->GetSolution(iPoint,iVar));
      solution_time_n_Saved[iPoint][iVar] = (solver_container[ZONE_0][INST_0][MESH_0][FLOW_SOL]->GetNodes()->GetSolution_time_n(iPoint,iVar));
      solution_time_n1_Saved[iPoint][iVar] = (solver_container[ZONE_0][INST_0][MESH_0][FLOW_SOL]->GetNodes()->GetSolution_time_n1(iPoint,iVar));
     // std::cout << " Point " << iPoint << " Variable 1 " << solution_Saved[iPoint][0] << std::endl;
    } 
    
    for (int iDim = 0; iDim < nDim; iDim++) 
    {
      //Save coordinates at last, current and next time step
      Coord_Saved[iPoint][iDim] =  (geometry_container[ZONE_0][INST_0][MESH_0]->nodes->GetCoord(iPoint))[iDim];
      Coord_n_Saved[iPoint][iDim] =  (geometry_container[ZONE_0][INST_0][MESH_0]->nodes->GetCoord_n(iPoint))[iDim];
      Coord_n1_Saved[iPoint][iDim] =  (geometry_container[ZONE_0][INST_0][MESH_0]->nodes->GetCoord_n1(iPoint))[iDim];
      Coord_p1_Saved[iPoint][iDim] =  (geometry_container[ZONE_0][INST_0][MESH_0]->nodes->GetCoord_p1(iPoint))[iDim];

      GridVel_Saved[iPoint][iDim] = (geometry_container[ZONE_0][INST_0][MESH_0]->nodes->GetGridVel(iPoint))[iDim];  
    } 
  }

  
    //---Recording GridVel_Grad using SU2 datatype---//

    
    GridVel_Grad = geometry_container[ZONE_0][INST_0][MESH_0]->nodes->GetGridVel_Grad();
    

  //Save wether simulation should be stopped after the current iteration
  StopCalc_savedState = *StopCalc;
  //Save the time step size
  dt_savedState = *dt;

  
  //Writing task has been fulfilled successfully
  solverInterface.markActionFulfilled(cowic);
}

void Precice ::saveOldStaticState( bool *StopCalc, double *dt)
{
  if (solverProcessIndex == 0)
  {
    std::cout << "Saving current state for implicit calculations" << std::endl;
  }

  /*---Begin loop over ALL grid points in the fluid domain---*/
  
  for (int iPoint = 0; iPoint < nPoint; iPoint++) 
  {
    for (int iVar = 0; iVar < nVar; iVar++) 
    {
      //Save solutions at last and current time step
      solution_Saved[iPoint][iVar] = (solver_container[ZONE_0][INST_0][MESH_0][FLOW_SOL]->GetNodes()->GetSolution(iPoint,iVar));
      
      //solution_time_n_Saved[iPoint][iVar] = (solver_container[ZONE_0][INST_0][MESH_0][FLOW_SOL]->GetNodes()->GetSolution_time_n(iPoint,iVar));
      
      //solution_time_n1_Saved[iPoint][iVar] = (solver_container[ZONE_0][INST_0][MESH_0][FLOW_SOL]->GetNodes()->GetSolution_time_n1(iPoint,iVar));
       
    } 
    
    for (int iDim = 0; iDim < nDim; iDim++) 
    {
      //Save coordinates at last, current and next time step
      Coord_Saved[iPoint][iDim] =  (geometry_container[ZONE_0][INST_0][MESH_0]->nodes->GetCoord(iPoint))[iDim];
      
      //Coord_n_Saved[iPoint][iDim] =  (geometry_container[ZONE_0][INST_0][MESH_0]->nodes->GetCoord_n(iPoint))[iDim];
      
     // Coord_n1_Saved[iPoint][iDim] =  (geometry_container[ZONE_0][INST_0][MESH_0]->nodes->GetCoord_n1(iPoint))[iDim];
     // Coord_p1_Saved[iPoint][iDim] =  (geometry_container[ZONE_0][INST_0][MESH_0]->nodes->GetCoord_p1(iPoint))[iDim];
    } 
  }

  
//Save wether simulation should be stopped after the current iteration
  StopCalc_savedState = *StopCalc;
  //Save the time step size
  dt_savedState = *dt;

  
  //Writing task has been fulfilled successfully
  solverInterface.markActionFulfilled(cowic);
}

void Precice::reloadOldState(bool *StopCalc, double *dt)
{
  if (solverProcessIndex == 0)
  {
    std::cout << "Reloading old states for implicit calculations" << std::endl;
  }
  
  
    
  for (int iPoint = 0; iPoint < nPoint; iPoint++)
  {
    solver_container[ZONE_0][INST_0][MESH_0][FLOW_SOL]->GetNodes()->SetSolution( iPoint, solution_Saved[iPoint]);
    solver_container[ZONE_0][INST_0][MESH_0][FLOW_SOL]->GetNodes()->Set_Solution_time_n(iPoint, solution_time_n_Saved[iPoint]);
    solver_container[ZONE_0][INST_0][MESH_0][FLOW_SOL]->GetNodes()->Set_Solution_time_n1( iPoint, solution_time_n1_Saved[iPoint]);

    //Reload coordinates at last, current and next time step
    
    geometry_container[ZONE_0][INST_0][MESH_0]->nodes->SetCoord(iPoint, Coord_n1_Saved[iPoint]);
    geometry_container[ZONE_0][INST_0][MESH_0]->nodes->SetCoord_n();
    geometry_container[ZONE_0][INST_0][MESH_0]->nodes->SetCoord_n1();
    geometry_container[ZONE_0][INST_0][MESH_0]->nodes->SetCoord(iPoint, Coord_n_Saved[iPoint]);
    geometry_container[ZONE_0][INST_0][MESH_0]->nodes->SetCoord_n();
    geometry_container[ZONE_0][INST_0][MESH_0]->nodes->SetCoord_p1(iPoint, Coord_p1_Saved[iPoint]);
    geometry_container[ZONE_0][INST_0][MESH_0]->nodes->SetCoord(iPoint, Coord_Saved[iPoint]);

    //Reload grid velocity
    geometry_container[ZONE_0][INST_0][MESH_0]->nodes->SetGridVel(iPoint, GridVel_Saved[iPoint]);
    
    
  }

  
  //--- Set the grid velocity gradient here---//
  
   geometry_container[ZONE_0][INST_0][MESH_0]->nodes->SetGridVel_Grad(GridVel_Grad);
   
  //Reload wether simulation should be stopped after current iteration
  *StopCalc = StopCalc_savedState;
  //Reload the time step size
  *dt = dt_savedState;
  //Reading task has been fulfilled successfully

  
  solverInterface.markActionFulfilled(coric);

}

void Precice::reloadOldStaticState(bool *StopCalc, double *dt)
{
  if (solverProcessIndex == 0)
  {
    std::cout << "Reloading old states for implicit calculations" << std::endl;
  }
  
  
    
  for (int iPoint = 0; iPoint < nPoint; iPoint++)
  {
    solver_container[ZONE_0][INST_0][MESH_0][FLOW_SOL]->GetNodes()->SetSolution( iPoint, solution_Saved[iPoint]);
   // solver_container[ZONE_0][INST_0][MESH_0][FLOW_SOL]->GetNodes()->Set_Solution_time_n(iPoint, solution_time_n_Saved[iPoint]);
   // solver_container[ZONE_0][INST_0][MESH_0][FLOW_SOL]->GetNodes()->Set_Solution_time_n1( iPoint, solution_time_n1_Saved[iPoint]);

    //Reload coordinates at last, current and next time step
    
   // geometry_container[ZONE_0][INST_0][MESH_0]->nodes->SetCoord(iPoint, Coord_n1_Saved[iPoint]);
  //  geometry_container[ZONE_0][INST_0][MESH_0]->nodes->SetCoord_n();
  //  geometry_container[ZONE_0][INST_0][MESH_0]->nodes->SetCoord_n1();
  //  geometry_container[ZONE_0][INST_0][MESH_0]->nodes->SetCoord(iPoint, Coord_n_Saved[iPoint]);
  //  geometry_container[ZONE_0][INST_0][MESH_0]->nodes->SetCoord_n();
  //  geometry_container[ZONE_0][INST_0][MESH_0]->nodes->SetCoord_p1(iPoint, Coord_p1_Saved[iPoint]);
    geometry_container[ZONE_0][INST_0][MESH_0]->nodes->SetCoord(iPoint, Coord_Saved[iPoint]);
        
  }
   
  //Reload wether simulation should be stopped after current iteration
  *StopCalc = StopCalc_savedState;
  //Reload the time step size
  *dt = dt_savedState;
  //Reading task has been fulfilled successfully

  
  solverInterface.markActionFulfilled(coric);

}

bool Precice::isCouplingOngoing()
{
  return solverInterface.isCouplingOngoing();
}

bool Precice::isActionRequired( const string& action )
{
  return solverInterface.isActionRequired(action);
}

const string& Precice::getCowic()
{
  return cowic;
}

const string& Precice::getCoric()
{
  return coric;
}

void Precice::finalize()
{
  solverInterface.finalize();
}



