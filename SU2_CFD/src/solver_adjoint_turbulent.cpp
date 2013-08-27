/*!
 * \file solution_adjoint_turbulent.cpp
 * \brief Main subrotuines for solving adjoint problems (Euler, Navier-Stokes, etc.).
 * \author Aerospace Design Laboratory (Stanford University) <http://su2.stanford.edu>.
 * \version 2.0.6
 *
 * Stanford University Unstructured (SU2) Code
 * Copyright (C) 2012 Aerospace Design Laboratory
 *
 * This program is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#include "../include/solver_structure.hpp"

CAdjTurbSolver::CAdjTurbSolver(void) : CSolver() {}

CAdjTurbSolver::CAdjTurbSolver(CGeometry *geometry, CConfig *config) : CSolver() {
	unsigned long iPoint;
	unsigned short nMarker, iDim, iVar;
  
	nDim = geometry->GetnDim();
	nMarker = config->GetnMarker_All();
	Gamma = config->GetGamma();
	Gamma_Minus_One = Gamma - 1.0;
  
	/*--- Dimension of the problem --> dependent of the turbulent model, only SA implemented ---*/
	switch (config->GetKind_Turb_Model()) {
		case SA : nVar = 1; break;
		case SST : nVar = 2; break;
	}
  
  nPoint = geometry->GetnPoint();
  nPointDomain = geometry->GetnPointDomain();
  
	Residual   = new double [nVar]; Residual_RMS = new double[nVar];
	Residual_i = new double [nVar]; Residual_j = new double [nVar];
	Residual_Max = new double [nVar]; Point_Max = new unsigned long[nVar];
  
	Solution   = new double [nVar];
	Solution_i = new double [nVar];
	Solution_j = new double [nVar];
  
	/*--- Define some auxiliar vector related with the geometry ---*/
	Vector_i = new double[nDim]; Vector_j = new double[nDim];
  
	/*--- Define some auxiliar vector related with the flow solution ---*/
	FlowSolution_i = new double [nDim+2]; FlowSolution_j = new double [nDim+2];
  
	/*--- Point to point Jacobians ---*/
	Jacobian_ii = new double* [nVar];
	Jacobian_ij = new double* [nVar];
	Jacobian_ji = new double* [nVar];
	Jacobian_jj = new double* [nVar];
	for (unsigned short iVar = 0; iVar < nVar; iVar++) {
		Jacobian_ii[iVar] = new double [nVar];
		Jacobian_ij[iVar] = new double [nVar];
		Jacobian_ji[iVar] = new double [nVar];
		Jacobian_jj[iVar] = new double [nVar];
	}
  
	/*--- Initialization of the structure of the whole Jacobian ---*/
	Jacobian.Initialize(nPoint, nPointDomain, nVar, nVar, geometry);
  Jacobian.SetValZero();
  LinSysSol.Initialize(nPoint, nPointDomain, nVar, 0.0);
  LinSysRes.Initialize(nPoint, nPointDomain, nVar, 0.0);
  
	/*--- Computation of gradients by least squares ---*/
	if (config->GetKind_Gradient_Method() == WEIGHTED_LEAST_SQUARES) {
		/*--- S matrix := inv(R)*traspose(inv(R)) ---*/
		Smatrix = new double* [nDim];
		for (iDim = 0; iDim < nDim; iDim++)
			Smatrix[iDim] = new double [nDim];
		/*--- c vector := transpose(WA)*(Wb) ---*/
		cvector = new double* [nVar+1];
		for (iVar = 0; iVar < nVar+1; iVar++)
			cvector[iVar] = new double [nDim];
	}
	
	/*--- Far-Field values and initizalization ---*/
	node = new CVariable* [nPoint];
	bool restart = config->GetRestart();
  
	if (!restart || geometry->GetFinestMGLevel() == false) {
		PsiNu_Inf = 0.0;
		for (iPoint = 0; iPoint < nPoint; iPoint++) {
			node[iPoint] = new CAdjTurbVariable(PsiNu_Inf, nDim, nVar, config);
		}
	}
	else {
		unsigned long index;
		double dull_val;
		string filename, AdjExt, text_line;
    ifstream restart_file;
    
    /*--- Restart the solution from file information ---*/
		string mesh_filename = config->GetSolution_AdjFileName();
		
		/*--- Change the name, depending of the objective function ---*/
		filename.assign(mesh_filename);
    unsigned short lastindex = filename.find_last_of(".");
    filename = filename.substr(0, lastindex);
    
		switch (config->GetKind_ObjFunc()) {
			case DRAG_COEFFICIENT: AdjExt = "_cd.dat"; break;
			case LIFT_COEFFICIENT: AdjExt = "_cl.dat"; break;
			case SIDEFORCE_COEFFICIENT: AdjExt = "_csf.dat"; break;
			case PRESSURE_COEFFICIENT: AdjExt = "_cp.dat"; break;
			case MOMENT_X_COEFFICIENT: AdjExt = "_cmx.dat"; break;
			case MOMENT_Y_COEFFICIENT: AdjExt = "_cmy.dat"; break;
			case MOMENT_Z_COEFFICIENT: AdjExt = "_cmz.dat"; break;
			case EFFICIENCY: AdjExt = "_eff.dat"; break;
			case EQUIVALENT_AREA: AdjExt = "_ea.dat"; break;
			case NEARFIELD_PRESSURE: AdjExt = "_nfp.dat"; break;
      case FORCE_X_COEFFICIENT: AdjExt = "_cfx.dat"; break;
			case FORCE_Y_COEFFICIENT: AdjExt = "_cfy.dat"; break;
			case FORCE_Z_COEFFICIENT: AdjExt = "_cfz.dat"; break;
      case THRUST_COEFFICIENT: AdjExt = "_ct.dat"; break;
      case TORQUE_COEFFICIENT: AdjExt = "_cq.dat"; break;
      case FIGURE_OF_MERIT: AdjExt = "_merit.dat"; break;
			case FREE_SURFACE: AdjExt = "_fs.dat"; break;
      case NOISE: AdjExt = "_fwh.dat"; break;
      case HEAT_LOAD: AdjExt = "_Q.dat"; break;
		}
		filename.append(AdjExt);
		restart_file.open(filename.data(), ios::in);
    
		/*--- In case there is no file ---*/
		if (restart_file.fail()) {
			cout << "There is no adjoint restart file!! " << filename.data() << "."<< endl;
			cout << "Press any key to exit..." << endl;
			cin.get(); exit(1);
		}
    
    /*--- In case this is a parallel simulation, we need to perform the
     Global2Local index transformation first. ---*/
    long *Global2Local;
    Global2Local = new long[geometry->GetGlobal_nPointDomain()];
    /*--- First, set all indices to a negative value by default ---*/
    for(iPoint = 0; iPoint < geometry->GetGlobal_nPointDomain(); iPoint++) {
      Global2Local[iPoint] = -1;
    }
    /*--- Now fill array with the transform values only for local points ---*/
    for(iPoint = 0; iPoint < nPointDomain; iPoint++) {
      Global2Local[geometry->node[iPoint]->GetGlobalIndex()] = iPoint;
    }
    
		/*--- Read all lines in the restart file ---*/
    long iPoint_Local; unsigned long iPoint_Global = 0;
    
    /*--- The first line is the header ---*/
    getline (restart_file, text_line);
    
    while (getline (restart_file,text_line)) {
			istringstream point_line(text_line);
      
      /*--- Retrieve local index. If this node from the restart file lives
       on a different processor, the value of iPoint_Local will be -1.
       Otherwise, the local index for this node on the current processor
       will be returned and used to instantiate the vars. ---*/
      iPoint_Local = Global2Local[iPoint_Global];
      if (iPoint_Local >= 0) {
        if (nDim == 2) point_line >> index >> dull_val >> dull_val >> dull_val >> dull_val >> Solution[0];
        if (nDim == 3) point_line >> index >> dull_val >> dull_val >> dull_val >> dull_val >> dull_val >> Solution[0];        
        node[iPoint_Local] = new CAdjTurbVariable(Solution[0], nDim, nVar, config);
      }
      iPoint_Global++;
    }
    
    /*--- Instantiate the variable class with an arbitrary solution
     at any halo/periodic nodes. The initial solution can be arbitrary,
     because a send/recv is performed immediately in the solver. ---*/
    for(iPoint = nPointDomain; iPoint < nPoint; iPoint++) {
      node[iPoint] = new CAdjTurbVariable(Solution[0], nDim, nVar, config);
    }
    
		/*--- Close the restart file ---*/
		restart_file.close();
    
    /*--- Free memory needed for the transformation ---*/
    delete [] Global2Local;
	}
  
  /*--- MPI solution ---*/
  Set_MPI_Solution(geometry, config);
  
}

CAdjTurbSolver::~CAdjTurbSolver(void) {
    
	for (unsigned short iVar = 0; iVar < nVar; iVar++) {
		delete [] Jacobian_ii[iVar];
		delete [] Jacobian_ij[iVar];
    delete [] Jacobian_ji[iVar];
		delete [] Jacobian_jj[iVar];
	}
  
  delete [] Jacobian_ii;
  delete [] Jacobian_ij;
  delete [] Jacobian_ji;
  delete [] Jacobian_jj;
  
}

void CAdjTurbSolver::Set_MPI_Solution(CGeometry *geometry, CConfig *config) {
	unsigned short iVar, iMarker, MarkerS, MarkerR;
	unsigned long iVertex, iPoint, nVertexS, nVertexR, nBufferS_Vector, nBufferR_Vector, nBufferS_Scalar, nBufferR_Scalar;
	double *Buffer_Receive_U = NULL, *Buffer_Send_U = NULL;
	int send_to, receive_from;
  
#ifndef NO_MPI
  MPI::Status status;
  MPI::Request send_request, recv_request;
#endif
  
	for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    
		if ((config->GetMarker_All_Boundary(iMarker) == SEND_RECEIVE) &&
        (config->GetMarker_All_SendRecv(iMarker) > 0)) {
			
			MarkerS = iMarker;  MarkerR = iMarker+1;
      
      send_to = config->GetMarker_All_SendRecv(MarkerS)-1;
			receive_from = abs(config->GetMarker_All_SendRecv(MarkerR))-1;
			
			nVertexS = geometry->nVertex[MarkerS];  nVertexR = geometry->nVertex[MarkerR];
			nBufferS_Vector = nVertexS*nVar;        nBufferR_Vector = nVertexR*nVar;
      nBufferS_Scalar = nVertexS;             nBufferR_Scalar = nVertexR;
      
      /*--- Allocate Receive and send buffers  ---*/
      Buffer_Receive_U = new double [nBufferR_Vector];
      Buffer_Send_U = new double[nBufferS_Vector];
            
      /*--- Copy the solution that should be sended ---*/
      for (iVertex = 0; iVertex < nVertexS; iVertex++) {
        iPoint = geometry->vertex[MarkerS][iVertex]->GetNode();
        for (iVar = 0; iVar < nVar; iVar++)
          Buffer_Send_U[iVar*nVertexS+iVertex] = node[iPoint]->GetSolution(iVar);
      }
      
#ifndef NO_MPI
      
      //      /*--- Send/Receive using non-blocking communications ---*/
      //      send_request = MPI::COMM_WORLD.Isend(Buffer_Send_U, nBufferS_Vector, MPI::DOUBLE, 0, send_to);
      //      recv_request = MPI::COMM_WORLD.Irecv(Buffer_Receive_U, nBufferR_Vector, MPI::DOUBLE, 0, receive_from);
      //      send_request.Wait(status);
      //      recv_request.Wait(status);
      
      /*--- Send/Receive information using Sendrecv ---*/
      MPI::COMM_WORLD.Sendrecv(Buffer_Send_U, nBufferS_Vector, MPI::DOUBLE, send_to, 0,
                               Buffer_Receive_U, nBufferR_Vector, MPI::DOUBLE, receive_from, 0);
      
#else
      
      /*--- Receive information without MPI ---*/
      for (iVertex = 0; iVertex < nVertexR; iVertex++) {
        iPoint = geometry->vertex[MarkerR][iVertex]->GetNode();
        for (iVar = 0; iVar < nVar; iVar++)
          Buffer_Receive_U[iVar*nVertexR+iVertex] = Buffer_Send_U[iVar*nVertexR+iVertex];
      }
      
#endif
      
      /*--- Deallocate send buffer ---*/
      delete [] Buffer_Send_U;
      
      /*--- Do the coordinate transformation ---*/
      for (iVertex = 0; iVertex < nVertexR; iVertex++) {
        
        /*--- Find point and its type of transformation ---*/
        iPoint = geometry->vertex[MarkerR][iVertex]->GetNode();
        
        /*--- Copy conservative variables. ---*/
        for (iVar = 0; iVar < nVar; iVar++)
          node[iPoint]->SetSolution(iVar, Buffer_Receive_U[iVar*nVertexR+iVertex]);
        
      }
      
      /*--- Deallocate receive buffer ---*/
      delete [] Buffer_Receive_U;
      
    }
    
	}
  
}

void CAdjTurbSolver::Set_MPI_Solution_Old(CGeometry *geometry, CConfig *config) {
	unsigned short iVar, iMarker, MarkerS, MarkerR;
	unsigned long iVertex, iPoint, nVertexS, nVertexR, nBufferS_Vector, nBufferR_Vector;
	double *Buffer_Receive_U = NULL, *Buffer_Send_U = NULL;
	int send_to, receive_from;
  
#ifndef NO_MPI
  MPI::Status status;
  MPI::Request send_request, recv_request;
#endif
  
	for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    
		if ((config->GetMarker_All_Boundary(iMarker) == SEND_RECEIVE) &&
        (config->GetMarker_All_SendRecv(iMarker) > 0)) {
			
			MarkerS = iMarker;  MarkerR = iMarker+1;
      
      send_to = config->GetMarker_All_SendRecv(MarkerS)-1;
			receive_from = abs(config->GetMarker_All_SendRecv(MarkerR))-1;
			
			nVertexS = geometry->nVertex[MarkerS];  nVertexR = geometry->nVertex[MarkerR];
			nBufferS_Vector = nVertexS*nVar;        nBufferR_Vector = nVertexR*nVar;
      
      /*--- Allocate Receive and send buffers  ---*/
      Buffer_Receive_U = new double [nBufferR_Vector];
      Buffer_Send_U = new double[nBufferS_Vector];
      
      /*--- Copy the solution old that should be sended ---*/
      for (iVertex = 0; iVertex < nVertexS; iVertex++) {
        iPoint = geometry->vertex[MarkerS][iVertex]->GetNode();
        for (iVar = 0; iVar < nVar; iVar++)
          Buffer_Send_U[iVar*nVertexS+iVertex] = node[iPoint]->GetSolution_Old(iVar);
      }
      
#ifndef NO_MPI
      
      //      /*--- Send/Receive using non-blocking communications ---*/
      //      send_request = MPI::COMM_WORLD.Isend(Buffer_Send_U, nBufferS_Vector, MPI::DOUBLE, 0, send_to);
      //      recv_request = MPI::COMM_WORLD.Irecv(Buffer_Receive_U, nBufferR_Vector, MPI::DOUBLE, 0, receive_from);
      //      send_request.Wait(status);
      //      recv_request.Wait(status);
      
      /*--- Send/Receive information using Sendrecv ---*/
      MPI::COMM_WORLD.Sendrecv(Buffer_Send_U, nBufferS_Vector, MPI::DOUBLE, send_to, 0,
                               Buffer_Receive_U, nBufferR_Vector, MPI::DOUBLE, receive_from, 0);
      
#else
      
      /*--- Receive information without MPI ---*/
      for (iVertex = 0; iVertex < nVertexR; iVertex++) {
        iPoint = geometry->vertex[MarkerR][iVertex]->GetNode();
        for (iVar = 0; iVar < nVar; iVar++)
          Buffer_Receive_U[iVar*nVertexR+iVertex] = Buffer_Send_U[iVar*nVertexR+iVertex];
      }
      
#endif
      
      /*--- Deallocate send buffer ---*/
      delete [] Buffer_Send_U;
      
      /*--- Do the coordinate transformation ---*/
      for (iVertex = 0; iVertex < nVertexR; iVertex++) {
        
        /*--- Find point and its type of transformation ---*/
        iPoint = geometry->vertex[MarkerR][iVertex]->GetNode();
        
        /*--- Copy transformed conserved variables back into buffer. ---*/
        for (iVar = 0; iVar < nVar; iVar++)
          node[iPoint]->SetSolution_Old(iVar, Buffer_Receive_U[iVar*nVertexR+iVertex]);
        
      }
      
      /*--- Deallocate receive buffer ---*/
      delete [] Buffer_Receive_U;
      
    }
    
	}
}

void CAdjTurbSolver::Set_MPI_Solution_Gradient(CGeometry *geometry, CConfig *config) {
	unsigned short iVar, iDim, iMarker, iPeriodic_Index, MarkerS, MarkerR;
	unsigned long iVertex, iPoint, nVertexS, nVertexR, nBufferS_Vector, nBufferR_Vector;
	double rotMatrix[3][3], *angles, theta, cosTheta, sinTheta, phi, cosPhi, sinPhi, psi, cosPsi, sinPsi,
  *Buffer_Receive_Gradient = NULL, *Buffer_Send_Gradient = NULL;
	int send_to, receive_from;
  
#ifndef NO_MPI
  MPI::Status status;
  MPI::Request send_request, recv_request;
#endif
  
  double **Gradient = new double* [nVar];
  for (iVar = 0; iVar < nVar; iVar++)
    Gradient[iVar] = new double[nDim];
  
	for (iMarker = 0; iMarker < config->GetnMarker_All(); iMarker++) {
    
		if ((config->GetMarker_All_Boundary(iMarker) == SEND_RECEIVE) &&
        (config->GetMarker_All_SendRecv(iMarker) > 0)) {
			
			MarkerS = iMarker;  MarkerR = iMarker+1;
      
      send_to = config->GetMarker_All_SendRecv(MarkerS)-1;
			receive_from = abs(config->GetMarker_All_SendRecv(MarkerR))-1;
			
			nVertexS = geometry->nVertex[MarkerS];  nVertexR = geometry->nVertex[MarkerR];
			nBufferS_Vector = nVertexS*nVar*nDim;        nBufferR_Vector = nVertexR*nVar*nDim;
      
      /*--- Allocate Receive and send buffers  ---*/
      Buffer_Receive_Gradient = new double [nBufferR_Vector];
      Buffer_Send_Gradient = new double[nBufferS_Vector];
      
      /*--- Copy the solution old that should be sended ---*/
      for (iVertex = 0; iVertex < nVertexS; iVertex++) {
        iPoint = geometry->vertex[MarkerS][iVertex]->GetNode();
        for (iVar = 0; iVar < nVar; iVar++)
          for (iDim = 0; iDim < nDim; iDim++)
            Buffer_Send_Gradient[iDim*nVar*nVertexS+iVar*nVertexS+iVertex] = node[iPoint]->GetGradient(iVar, iDim);
      }
      
#ifndef NO_MPI
      
      //      /*--- Send/Receive using non-blocking communications ---*/
      //      send_request = MPI::COMM_WORLD.Isend(Buffer_Send_Gradient, nBufferS_Vector, MPI::DOUBLE, 0, send_to);
      //      recv_request = MPI::COMM_WORLD.Irecv(Buffer_Receive_Gradient, nBufferR_Vector, MPI::DOUBLE, 0, receive_from);
      //      send_request.Wait(status);
      //      recv_request.Wait(status);
      
      /*--- Send/Receive information using Sendrecv ---*/
      MPI::COMM_WORLD.Sendrecv(Buffer_Send_Gradient, nBufferS_Vector, MPI::DOUBLE, send_to, 0,
                               Buffer_Receive_Gradient, nBufferR_Vector, MPI::DOUBLE, receive_from, 0);
      
#else
      
      /*--- Receive information without MPI ---*/
      for (iVertex = 0; iVertex < nVertexR; iVertex++) {
        iPoint = geometry->vertex[MarkerR][iVertex]->GetNode();
        for (iVar = 0; iVar < nVar; iVar++)
          for (iDim = 0; iDim < nDim; iDim++)
            Buffer_Receive_Gradient[iDim*nVar*nVertexR+iVar*nVertexR+iVertex] = Buffer_Send_Gradient[iDim*nVar*nVertexR+iVar*nVertexR+iVertex];
      }
      
#endif
      
      /*--- Deallocate send buffer ---*/
      delete [] Buffer_Send_Gradient;
      
      /*--- Do the coordinate transformation ---*/
      for (iVertex = 0; iVertex < nVertexR; iVertex++) {
        
        /*--- Find point and its type of transformation ---*/
        iPoint = geometry->vertex[MarkerR][iVertex]->GetNode();
        iPeriodic_Index = geometry->vertex[MarkerR][iVertex]->GetRotation_Type();
        
        /*--- Retrieve the supplied periodic information. ---*/
        angles = config->GetPeriodicRotation(iPeriodic_Index);
        
        /*--- Store angles separately for clarity. ---*/
        theta    = angles[0];   phi    = angles[1];     psi    = angles[2];
        cosTheta = cos(theta);  cosPhi = cos(phi);      cosPsi = cos(psi);
        sinTheta = sin(theta);  sinPhi = sin(phi);      sinPsi = sin(psi);
        
        /*--- Compute the rotation matrix. Note that the implicit
         ordering is rotation about the x-axis, y-axis,
         then z-axis. Note that this is the transpose of the matrix
         used during the preprocessing stage. ---*/
        rotMatrix[0][0] = cosPhi*cosPsi;    rotMatrix[1][0] = sinTheta*sinPhi*cosPsi - cosTheta*sinPsi;     rotMatrix[2][0] = cosTheta*sinPhi*cosPsi + sinTheta*sinPsi;
        rotMatrix[0][1] = cosPhi*sinPsi;    rotMatrix[1][1] = sinTheta*sinPhi*sinPsi + cosTheta*cosPsi;     rotMatrix[2][1] = cosTheta*sinPhi*sinPsi - sinTheta*cosPsi;
        rotMatrix[0][2] = -sinPhi;          rotMatrix[1][2] = sinTheta*cosPhi;                              rotMatrix[2][2] = cosTheta*cosPhi;
        
        /*--- Copy conserved variables before performing transformation. ---*/
        for (iVar = 0; iVar < nVar; iVar++)
          for (iDim = 0; iDim < nDim; iDim++)
            Gradient[iVar][iDim] = Buffer_Receive_Gradient[iDim*nVar*nVertexR+iVar*nVertexR+iVertex];
        
        /*--- Need to rotate the gradients for all conserved variables. ---*/
        for (iVar = 0; iVar < nVar; iVar++) {
          if (nDim == 2) {
            Gradient[iVar][0] = rotMatrix[0][0]*Buffer_Receive_Gradient[0*nVar*nVertexR+iVar*nVertexR+iVertex] + rotMatrix[0][1]*Buffer_Receive_Gradient[1*nVar*nVertexR+iVar*nVertexR+iVertex];
            Gradient[iVar][1] = rotMatrix[1][0]*Buffer_Receive_Gradient[0*nVar*nVertexR+iVar*nVertexR+iVertex] + rotMatrix[1][1]*Buffer_Receive_Gradient[1*nVar*nVertexR+iVar*nVertexR+iVertex];
          }
          else {
            Gradient[iVar][0] = rotMatrix[0][0]*Buffer_Receive_Gradient[0*nVar*nVertexR+iVar*nVertexR+iVertex] + rotMatrix[0][1]*Buffer_Receive_Gradient[1*nVar*nVertexR+iVar*nVertexR+iVertex] + rotMatrix[0][2]*Buffer_Receive_Gradient[2*nVar*nVertexR+iVar*nVertexR+iVertex];
            Gradient[iVar][1] = rotMatrix[1][0]*Buffer_Receive_Gradient[0*nVar*nVertexR+iVar*nVertexR+iVertex] + rotMatrix[1][1]*Buffer_Receive_Gradient[1*nVar*nVertexR+iVar*nVertexR+iVertex] + rotMatrix[1][2]*Buffer_Receive_Gradient[2*nVar*nVertexR+iVar*nVertexR+iVertex];
            Gradient[iVar][2] = rotMatrix[2][0]*Buffer_Receive_Gradient[0*nVar*nVertexR+iVar*nVertexR+iVertex] + rotMatrix[2][1]*Buffer_Receive_Gradient[1*nVar*nVertexR+iVar*nVertexR+iVertex] + rotMatrix[2][2]*Buffer_Receive_Gradient[2*nVar*nVertexR+iVar*nVertexR+iVertex];
          }
        }
        
        /*--- Store the received information ---*/
        for (iVar = 0; iVar < nVar; iVar++)
          for (iDim = 0; iDim < nDim; iDim++)
            node[iPoint]->SetGradient(iVar, iDim, Gradient[iVar][iDim]);
        
      }
      
      /*--- Deallocate receive buffer ---*/
      delete [] Buffer_Receive_Gradient;
      
    }
    
	}
  
  for (iVar = 0; iVar < nVar; iVar++)
    delete [] Gradient[iVar];
  delete [] Gradient;
  
}

void CAdjTurbSolver::BC_HeatFlux_Wall(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics, CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {
  
	unsigned long iPoint, iVertex;
  
	for (iVertex = 0; iVertex<geometry->nVertex[val_marker]; iVertex++) {
		iPoint = geometry->vertex[val_marker][iVertex]->GetNode();
    
    /*--- Check if the node belongs to the domain (i.e, not a halo node) ---*/
		if (geometry->node[iPoint]->GetDomain()) {
      
      Solution[0] = 0.0;
      
      /*--- Set the solution values and zero the residual ---*/
      node[iPoint]->SetSolution_Old(Solution);
      LinSysRes.SetBlock_Zero(iPoint);
      
      /*--- Change rows of the Jacobian (includes 1 in the diagonal) ---*/
      Jacobian.DeleteValsRowi(iPoint);
      
    }
  }
  
}

void CAdjTurbSolver::BC_Isothermal_Wall(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics, CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {
  
	unsigned long iPoint, iVertex;
  
	for (iVertex = 0; iVertex<geometry->nVertex[val_marker]; iVertex++) {
		iPoint = geometry->vertex[val_marker][iVertex]->GetNode();
    
    /*--- Check if the node belongs to the domain (i.e, not a halo node) ---*/
		if (geometry->node[iPoint]->GetDomain()) {
      
      Solution[0] = 0.0;
      
      /*--- Set the solution values and zero the residual ---*/
      node[iPoint]->SetSolution_Old(Solution);
      LinSysRes.SetBlock_Zero(iPoint);
      
      /*--- Change rows of the Jacobian (includes 1 in the diagonal) ---*/
      Jacobian.DeleteValsRowi(iPoint);
      
    }
  }
  
}

void CAdjTurbSolver::BC_Far_Field(CGeometry *geometry, CSolver **solver_container, CNumerics *conv_numerics, CNumerics *visc_numerics, CConfig *config, unsigned short val_marker) {
	unsigned long iPoint, iVertex;
  
  for (iVertex = 0; iVertex < geometry->nVertex[val_marker]; iVertex++) {
    
    iPoint = geometry->vertex[val_marker][iVertex]->GetNode();
    
    /*--- Set Normal ---*/
    conv_numerics->SetNormal(geometry->vertex[val_marker][iVertex]->GetNormal());

    /*--- Set Conservative variables (for convection) ---*/
    double* U_i = solver_container[FLOW_SOL]->node[iPoint]->GetSolution();
    conv_numerics->SetConservative(U_i, NULL);
    
    /*--- Turbulent adjoint variables w/o reconstruction ---*/
    double* TurbPsi_i = node[iPoint]->GetSolution();
    conv_numerics->SetTurbAdjointVar(TurbPsi_i, NULL);
    
    /*--- Add Residuals and Jacobians ---*/
    conv_numerics->ComputeResidual(Residual, Jacobian_ii, NULL, config);
    LinSysRes.AddBlock(iPoint, Residual);
    Jacobian.AddBlock(iPoint, iPoint, Jacobian_ii);
    
	}
  
}

void CAdjTurbSolver::Preprocessing(CGeometry *geometry, CSolver **solver_container, CConfig *config, unsigned short iMesh, unsigned short iRKStep, unsigned short RunTime_EqSystem) {
  unsigned long iPoint;
  
	for (iPoint = 0; iPoint < nPoint; iPoint++) {
    
  /*--- Initialize the residual vector ---*/
		LinSysRes.SetBlock_Zero(iPoint);
    
  }
  
  
    /*--- Initialize the jacobian matrices ---*/
  Jacobian.SetValZero();
  
  /*--- Gradient of the adjoint turbulent variables ---*/
	if (config->GetKind_Gradient_Method() == GREEN_GAUSS) SetSolution_Gradient_GG(geometry, config);
	if (config->GetKind_Gradient_Method() == WEIGHTED_LEAST_SQUARES) SetSolution_Gradient_LS(geometry, config);
  
  /*--- Gradient of the turbulent variables ---*/
  if (config->GetKind_Gradient_Method() == GREEN_GAUSS) solver_container[TURB_SOL]->SetSolution_Gradient_GG(geometry, config);
	if (config->GetKind_Gradient_Method() == WEIGHTED_LEAST_SQUARES) solver_container[TURB_SOL]->SetSolution_Gradient_LS(geometry, config);
  
}

void CAdjTurbSolver::Upwind_Residual(CGeometry *geometry, CSolver **solver_container, CNumerics *numerics, CConfig *config, unsigned short iMesh) {
  
	unsigned long iEdge, iPoint, jPoint;
	double *U_i, *U_j, *TurbPsi_i, *TurbPsi_j, **TurbVar_Grad_i, **TurbVar_Grad_j;
//  double *Limiter_i = NULL, *Limiter_j = NULL, **Gradient_i, **Gradient_j, Project_Grad_i, Project_Grad_j;
//	unsigned short iDim, iVar;
  
	bool high_order_diss = (config->GetKind_Upwind_AdjTurb() == SCALAR_UPWIND_2ND);
	bool limiter = (config->GetKind_SlopeLimit() != NONE);
  
	if (high_order_diss) {
		if (config->GetKind_Gradient_Method() == GREEN_GAUSS) SetSolution_Gradient_GG(geometry, config);
		if (config->GetKind_Gradient_Method() == WEIGHTED_LEAST_SQUARES) SetSolution_Gradient_LS(geometry, config);
		if (limiter) SetSolution_Limiter(geometry, config);
	}
  
  for (iEdge = 0; iEdge < geometry->GetnEdge(); iEdge++) {
    
    /*--- Points in edge ---*/
    iPoint = geometry->edge[iEdge]->GetNode(0);
    jPoint = geometry->edge[iEdge]->GetNode(1);
    
    /*--- Conservative variables w/o reconstruction ---*/
    U_i = solver_container[FLOW_SOL]->node[iPoint]->GetSolution();
    U_j = solver_container[FLOW_SOL]->node[jPoint]->GetSolution();
    numerics->SetConservative(U_i, U_j);
    
    /*--- Set normal vectors and length ---*/
    numerics->SetNormal(geometry->edge[iEdge]->GetNormal());
    
    /*--- Turbulent adjoint variables w/o reconstruction ---*/
    TurbPsi_i = node[iPoint]->GetSolution();
    TurbPsi_j = node[jPoint]->GetSolution();
    numerics->SetTurbAdjointVar(TurbPsi_i, TurbPsi_j);
    
    /*--- Gradient of turbulent variables w/o reconstruction ---*/
    TurbVar_Grad_i = solver_container[TURB_SOL]->node[iPoint]->GetGradient();
    TurbVar_Grad_j = solver_container[TURB_SOL]->node[jPoint]->GetGradient();
    numerics->SetTurbVarGradient(TurbVar_Grad_i, TurbVar_Grad_j);
    
//    if (high_order_diss) {
//      
//      /*--- Conservative solution using gradient reconstruction ---*/
//      for (iDim = 0; iDim < nDim; iDim++) {
//        Vector_i[iDim] = 0.5*(geometry->node[jPoint]->GetCoord(iDim) - geometry->node[iPoint]->GetCoord(iDim));
//        Vector_j[iDim] = 0.5*(geometry->node[iPoint]->GetCoord(iDim) - geometry->node[jPoint]->GetCoord(iDim));
//      }
//      Gradient_i = solver_container[FLOW_SOL]->node[iPoint]->GetGradient();
//      Gradient_j = solver_container[FLOW_SOL]->node[jPoint]->GetGradient();
//      for (iVar = 0; iVar < solver_container[FLOW_SOL]->GetnVar(); iVar++) {
//        Project_Grad_i = 0; Project_Grad_j = 0;
//        for (iDim = 0; iDim < nDim; iDim++) {
//          Project_Grad_i += Vector_i[iDim]*Gradient_i[iVar][iDim];
//          Project_Grad_j += Vector_j[iDim]*Gradient_j[iVar][iDim];
//        }
//        FlowSolution_i[iVar] = U_i[iVar] + Project_Grad_i;
//        FlowSolution_j[iVar] = U_j[iVar] + Project_Grad_j;
//      }
//      numerics->SetConservative(FlowSolution_i, FlowSolution_j);
//      
//      /*--- Adjoint turbulent variables using gradient reconstruction ---*/
//      Gradient_i = node[iPoint]->GetGradient(); Gradient_j = node[jPoint]->GetGradient();
//      if (limiter) { Limiter_i = node[iPoint]->GetLimiter(); Limiter_j = node[jPoint]->GetLimiter(); }
//      for (iVar = 0; iVar < nVar; iVar++) {
//        Project_Grad_i = 0; Project_Grad_j = 0;
//        for (iDim = 0; iDim < nDim; iDim++) {
//          Project_Grad_i += Vector_i[iDim]*Gradient_i[iVar][iDim];
//          Project_Grad_j += Vector_j[iDim]*Gradient_j[iVar][iDim];
//        }
//        if (limiter) {
//          Solution_i[iVar] = TurbPsi_i[iVar] + Project_Grad_i*Limiter_i[iVar];
//          Solution_j[iVar] = TurbPsi_j[iVar] + Project_Grad_j*Limiter_j[iVar];
//        }
//        else {
//          Solution_i[iVar] = TurbPsi_i[iVar] + Project_Grad_i;
//          Solution_j[iVar] = TurbPsi_j[iVar] + Project_Grad_j;
//        }
//      }
//      numerics->SetTurbVar(Solution_i, Solution_j);
//      
//    }
    
    /*--- Set normal vectors and length ---*/
    numerics->SetNormal(geometry->edge[iEdge]->GetNormal());
    
    numerics->ComputeResidual(Residual_i, Residual_j, Jacobian_ii, Jacobian_ij, Jacobian_ji, Jacobian_jj, config);
    
    /*--- Add and Subtract Residual ---*/
    LinSysRes.AddBlock(iPoint, Residual_i);
    LinSysRes.AddBlock(jPoint, Residual_j);
    Jacobian.AddBlock(iPoint,iPoint,Jacobian_ii);
    Jacobian.AddBlock(iPoint,jPoint,Jacobian_ij);
    Jacobian.AddBlock(jPoint,iPoint,Jacobian_ji);
    Jacobian.AddBlock(jPoint,jPoint,Jacobian_jj);
    
  }
  
}

void CAdjTurbSolver::Viscous_Residual(CGeometry *geometry, CSolver **solver_container, CNumerics *numerics, CConfig *config,
                                      unsigned short iMesh, unsigned short iRKStep) {
	unsigned long iEdge, iPoint, jPoint;
	double *Coord_i, *Coord_j;
  
  for (iEdge = 0; iEdge < geometry->GetnEdge(); iEdge++) {
    
    /*--- Points in edge ---*/
    iPoint = geometry->edge[iEdge]->GetNode(0);
    jPoint = geometry->edge[iEdge]->GetNode(1);
    
    /*--- Points coordinates, and set normal vectors and length ---*/
    Coord_i = geometry->node[iPoint]->GetCoord();
    Coord_j = geometry->node[jPoint]->GetCoord();
    numerics->SetCoord(Coord_i, Coord_j);
    numerics->SetNormal(geometry->edge[iEdge]->GetNormal());
    
    /*--- Conservative variables w/o reconstruction, turbulent variables w/o reconstruction,
     and turbulent adjoint variables w/o reconstruction ---*/
    numerics->SetConservative(solver_container[FLOW_SOL]->node[iPoint]->GetSolution(), solver_container[FLOW_SOL]->node[jPoint]->GetSolution());
    numerics->SetTurbVar(solver_container[TURB_SOL]->node[iPoint]->GetSolution(), solver_container[TURB_SOL]->node[jPoint]->GetSolution());
    numerics->SetTurbAdjointVar(node[iPoint]->GetSolution(), node[jPoint]->GetSolution());
    
    /*--- Viscosity ---*/
    numerics->SetLaminarViscosity(solver_container[FLOW_SOL]->node[iPoint]->GetLaminarViscosity(),
                                  solver_container[FLOW_SOL]->node[jPoint]->GetLaminarViscosity());
    
    /*--- Turbulent adjoint variables w/o reconstruction ---*/
    numerics->SetTurbAdjointGradient(node[iPoint]->GetGradient(), node[jPoint]->GetGradient());
    
    /*--- Compute residual in a non-conservative way, and update ---*/
    numerics->ComputeResidual(Residual_i, Residual_j, Jacobian_ii, Jacobian_ij, Jacobian_ji, Jacobian_jj, config);
    
    /*--- Update adjoint viscous residual ---*/
    LinSysRes.AddBlock(iPoint, Residual_i);
    LinSysRes.AddBlock(jPoint, Residual_j);
    
    Jacobian.AddBlock(iPoint, iPoint, Jacobian_ii);
    Jacobian.AddBlock(iPoint, jPoint, Jacobian_ij);
    Jacobian.AddBlock(jPoint, iPoint, Jacobian_ji);
    Jacobian.AddBlock(jPoint, jPoint, Jacobian_jj);
    
  }
  
}

void CAdjTurbSolver::Source_Residual(CGeometry *geometry, CSolver **solver_container, CNumerics *numerics, CNumerics *second_numerics, CConfig *config, unsigned short iMesh) {
	unsigned long iPoint, jPoint, iEdge;
	double *U_i, **GradPrimVar_i, *TurbVar_i;
	double **TurbVar_Grad_i, **TurbVar_Grad_j, *TurbPsi_i, *TurbPsi_j, **PsiVar_Grad_i; // Gradients
  
  /*--- Piecewise source term ---*/
	for (iPoint = 0; iPoint < nPointDomain; iPoint++) {
        
    /*--- Conservative variables w/o reconstruction ---*/
    U_i = solver_container[FLOW_SOL]->node[iPoint]->GetSolution();
    numerics->SetConservative(U_i, NULL);
    
    /*--- Gradient of primitive variables w/o reconstruction ---*/
    GradPrimVar_i = solver_container[FLOW_SOL]->node[iPoint]->GetGradient_Primitive();
    numerics->SetPrimVarGradient(GradPrimVar_i, NULL);
    
    /*--- Laminar viscosity of the fluid ---*/
    numerics->SetLaminarViscosity(solver_container[FLOW_SOL]->node[iPoint]->GetLaminarViscosity(), 0.0);
    
    /*--- Turbulent variables w/o reconstruction ---*/
    TurbVar_i = solver_container[TURB_SOL]->node[iPoint]->GetSolution();
    numerics->SetTurbVar(TurbVar_i, NULL);
    
    /*--- Gradient of Turbulent Variables w/o reconstruction ---*/
    TurbVar_Grad_i = solver_container[TURB_SOL]->node[iPoint]->GetGradient();
    numerics->SetTurbVarGradient(TurbVar_Grad_i, NULL);
    
    /*--- Turbulent adjoint variables w/o reconstruction ---*/
    TurbPsi_i = node[iPoint]->GetSolution();
    numerics->SetTurbAdjointVar(TurbPsi_i, NULL);
    
    /*--- Gradient of Adjoint flow variables w/o reconstruction
     (for non-conservative terms depending on gradients of flow adjoint vars.) ---*/
    PsiVar_Grad_i = solver_container[ADJFLOW_SOL]->node[iPoint]->GetGradient();
    numerics->SetAdjointVarGradient(PsiVar_Grad_i, NULL);
    
    /*--- Set volume and distances to the surface ---*/
    numerics->SetVolume(geometry->node[iPoint]->GetVolume());
    numerics->SetDistance(geometry->node[iPoint]->GetWall_Distance(), 0.0);
    
    /*--- Add and Subtract Residual ---*/
    numerics->ComputeResidual(Residual, Jacobian_ii, NULL, config);
    LinSysRes.AddBlock(iPoint, Residual);
    Jacobian.AddBlock(iPoint, iPoint, Jacobian_ii);
    
	}
  
//  /*--- Conservative Source Term ---*/
//  for (iEdge = 0; iEdge < geometry->GetnEdge(); iEdge++) {
//    
//    /*--- Points in edge ---*/
//    iPoint = geometry->edge[iEdge]->GetNode(0);
//    jPoint = geometry->edge[iEdge]->GetNode(1);
//    
//    /*--- Gradient of turbulent variables w/o reconstruction ---*/
//    TurbVar_Grad_i = solver_container[TURB_SOL]->node[iPoint]->GetGradient();
//    TurbVar_Grad_j = solver_container[TURB_SOL]->node[jPoint]->GetGradient();
//    second_numerics->SetTurbVarGradient(TurbVar_Grad_i, TurbVar_Grad_j);
//    
//    /*--- Turbulent adjoint variables w/o reconstruction ---*/
//    TurbPsi_i = node[iPoint]->GetSolution();
//    TurbPsi_j = node[jPoint]->GetSolution();
//    second_numerics->SetTurbAdjointVar(TurbPsi_i, TurbPsi_j);
//    
//    /*--- Set normal vectors and length ---*/
//    second_numerics->SetNormal(geometry->edge[iEdge]->GetNormal());
//    
//    /*--- Add and Subtract Residual ---*/
//    second_numerics->ComputeResidual(Residual, Jacobian_ii, Jacobian_jj, config);
//    LinSysRes.AddBlock(iPoint, Residual);
//    LinSysRes.SubtractBlock(jPoint, Residual);
//    Jacobian.AddBlock(iPoint,iPoint, Jacobian_ii);
//    Jacobian.AddBlock(iPoint,jPoint, Jacobian_jj);
//    Jacobian.SubtractBlock(jPoint, iPoint, Jacobian_ii);
//    Jacobian.SubtractBlock(jPoint, jPoint, Jacobian_jj);
//    
//  }
  
}

void CAdjTurbSolver::ImplicitEuler_Iteration(CGeometry *geometry, CSolver **solver_container, CConfig *config) {
	unsigned short iVar;
	unsigned long iPoint, total_index;
	double Delta, Vol;
  
	/*--- Set maximum residual to zero ---*/
	for (iVar = 0; iVar < nVar; iVar++) {
		SetRes_RMS(iVar, 0.0);
    SetRes_Max(iVar, 0.0, 0);
  }
  
	/*--- Build implicit system ---*/
	for (iPoint = 0; iPoint < nPoint; iPoint++) {
    
    /*--- Read the volume ---*/
		Vol = geometry->node[iPoint]->GetVolume();
    
		/*--- Modify matrix diagonal to assure diagonal dominance ---*/
		Delta = Vol / (config->GetAdjTurb_CFLRedCoeff()*solver_container[FLOW_SOL]->node[iPoint]->GetDelta_Time());
    
		Jacobian.AddVal2Diag(iPoint,Delta);
    
    /*--- Right hand side of the system (-Residual) and initial guess (x = 0) ---*/
		for (iVar = 0; iVar < nVar; iVar++) {
			total_index = iPoint*nVar+iVar;
			LinSysRes[total_index] = -LinSysRes[total_index];
			LinSysSol[total_index] = 0.0;
      AddRes_RMS(iVar, LinSysRes[total_index]*LinSysRes[total_index]);
      AddRes_Max(iVar, fabs(LinSysRes[total_index]), geometry->node[iPoint]->GetGlobalIndex());
		}
    
	}
  
  /*--- Initialize residual and solution at the ghost points ---*/
  for (iPoint = nPointDomain; iPoint < nPoint; iPoint++) {
    for (iVar = 0; iVar < nVar; iVar++) {
      total_index = iPoint*nVar + iVar;
      LinSysRes[total_index] = 0.0;
      LinSysSol[total_index] = 0.0;
    }
  }
  
	/*--- Solve the linear system (Krylov subspace methods) ---*/
  CMatrixVectorProduct* mat_vec = new CSysMatrixVectorProduct(Jacobian, geometry, config);
  
  CPreconditioner* precond = NULL;
  if (config->GetKind_Linear_Solver_Prec() == JACOBI) {
    Jacobian.BuildJacobiPreconditioner();
    precond = new CJacobiPreconditioner(Jacobian, geometry, config);
  }
  else if (config->GetKind_Linear_Solver_Prec() == LU_SGS) {
    precond = new CLU_SGSPreconditioner(Jacobian, geometry, config);
  }
  else if (config->GetKind_Linear_Solver_Prec() == LINELET) {
    Jacobian.BuildJacobiPreconditioner();
    Jacobian.BuildLineletPreconditioner(geometry, config);
    precond = new CLineletPreconditioner(Jacobian, geometry, config);
  }
  
  CSysSolve system;
  if (config->GetKind_Linear_Solver() == BCGSTAB)
    system.BCGSTAB(LinSysRes, LinSysSol, *mat_vec, *precond, config->GetAdjTurb_Linear_Error(),
                   config->GetAdjTurb_Linear_Iter(), false);
  else if (config->GetKind_Linear_Solver() == FGMRES)
    system.FGMRES(LinSysRes, LinSysSol, *mat_vec, *precond, config->GetAdjTurb_Linear_Error(),
                  config->GetAdjTurb_Linear_Iter(), false);
  
  delete mat_vec;
  delete precond;
  
	/*--- Update solution (system written in terms of increments) ---*/
	for (iPoint = 0; iPoint < nPointDomain; iPoint++) {
		for (iVar = 0; iVar < nVar; iVar++)
      node[iPoint]->AddSolution(iVar, config->GetLinear_Solver_Relax()*LinSysSol[iPoint*nVar+iVar]);
	}
  
  /*--- MPI solution ---*/
  Set_MPI_Solution(geometry, config);
  
  /*--- Compute the root mean square residual ---*/
  SetResidual_RMS(geometry, config);
  
}