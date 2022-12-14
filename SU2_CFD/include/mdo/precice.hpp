/*!
* \file precice.hpp
* \brief Header of adapter class for coupling SU2 with preCICE for FSI.
* \author Alexander Rusch
*/

#include "../SU2_CFD.hpp"
#include <string>
#include <stdlib.h>

class Precice {
private:
  //MPI-rank(=index) and size
  int solverProcessIndex, solverProcessSize;
  //Coupling interface object
  //SolverInterface solverInterface;
  //Mesh and boundary information
  //CGeometry**** geometry_container;
  //Solution information
  CSolver**** solver_container;
  //Configuration information
  CConfig** config_container;
  //Grid movement information
  CVolumetricMovement** grid_movement;

  //General preCICE-related Variables
  int nDim;
  unsigned long *vertexSize; //Number of nodes at the wet surface (for each wet surface)
  short *valueMarkerWet; //List of wet surface marker values
  int **vertexIDs;
  int *forceID;
  int *displDeltaID;
  double *forces;
  double *displacements;
  double *displacementDeltas;
  //const string& coric;
  //const string& cowic;
  bool processWorkingOnWetSurface;
  bool verbosityLevel_high;
  unsigned long globalNumberWetSurfaces; //Number of wet surfaces of the whole problem
  unsigned long localNumberWetSurfaces; //Number of wet surfaces, which this process is actually working on
  short *indexMarkerWetMappingLocalToGlobal; //Mapping relations for wet surfaces

    //Variables for storing the old state to reset to in case of an implicit simulation
  int nPoint; //Overall number of nodes of the problem
  int nVar; //Number of variables of the problem
  double **Coord_Saved, **Coord_n_Saved, **Coord_n1_Saved, **Coord_p1_Saved, **GridVel_Saved, ***GridVel_Grad_Saved;
  double dt_savedState;
  bool StopCalc_savedState;
  double **solution_Saved, **solution_time_n_Saved, **solution_time_n1_Saved;

  public:
  /*!
  * \brief Constructor of the class.
  * \param[in] configurationFileName - Name (with path) of the xml configuration
  *        file to be read.
  * \param[in] solverProcessIndex - Index of this solver Process.
  * \param[in] solverProcessSize - Overall number of solver Processes.
  * \param[in] geometry_container - Contains all geometric information of SU2.
  * \param[in] solver_container - Contains all solution and solver information of SU2.
  * \param[in] config_container - Contains all configuration information of SU2.
  * \param[in] grid_movement - Contains all grid movement information of SU2.
  */
  //Precice( const string& preciceConfigurationFileName, int solverProcessIndex, int solverProcessSize, CGeometry*** geometry_container, CSolver**** solver_container, CConfig** config_container, CVolumetricMovement** grid_movement );
  Precice( const string& preciceConfigurationFileName, int solverProcessIndex, int solverProcessSize, CSolver***** solver_container, CConfig** config_container, CVolumetricMovement*** grid_movement );

  /*!
  * \brief Destructor of the class.
  */
  ~Precice();

  /* Method to see if this adapter file is being called from SU2 */

  void check();

  

  

};

