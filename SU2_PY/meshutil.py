# This script performs four steps in sequence : 1) Open and read the displaced coordinates 2) Open and Read the existing mesh 3) Update mesh and write to file 4) Read
# the stress fields and 5) Write visualization files 
import numpy as np
import pandas as pd
import math
import sys 
import glob
import os

def MeshUpdate(Mesh_Name, Nnodes,Nelems,n):

    # Open the displacement file (Disp.csv)
    Up_CRD = pd.read_csv('updatedCOORD.csv',header = None, sep = ',', skiprows = 1, nrows=Nnodes)

    CRD = Up_CRD.to_numpy()

    # Get the node count from the Disp.csv file and see if all values are read 
    # Doing (Nnodes -1) since python begins reading from 0

    Node_Count_CRD = CRD[Nnodes-1,0]

    if ( Node_Count_CRD-Nnodes > 0.00001):
        print("Did not read all nodes from Disp.csv")
        print("Read from Disp.csv ", Node_Count_CRD)
        print("Actual number of nodes ", Nnodes)


    # Get the X displacement
    X_CRD = CRD[0:Nnodes,1]

    # Get the Y displacement 
    Y_CRD = CRD[0:Nnodes,2]

    # Get the Z displacement 
    Z_CRD = CRD[0:Nnodes,3]



    # Reshape all *_Disp arrays 
    X_CRD = X_CRD.reshape(Nnodes,1)
    Y_CRD = Y_CRD.reshape(Nnodes,1)
    Z_CRD = Z_CRD.reshape(Nnodes,1)

    #---------------------------------------------------------FINISHED READING ALL DISPLACEMENT DATA----------------------------------------------------------------------#


    Nodes = pd.read_csv(Mesh_Name,header = None, sep = ',', skiprows = 2, nrows=Nnodes)

    Coords= Nodes.to_numpy()

    # Check if correct number of nodes are read from the *.nam file 

    Node_Count_Msh = Coords[Nnodes-1,0]

    if ( Node_Count_Msh-Nnodes > 0.00001):
        print("Did not read all nodes from nam file")
        print("Read from *.msh ", Node_Count_Msh)
        print("Actual number of nodes ", Nnodes)

    # Get the Nodeid ( Needed for writing mesh update)

    Nodeid = Coords[0:Nnodes,0]
    Nodeid = Nodeid.reshape(Nnodes,1)    

    # Get the coordinates for all Nodes 

    X = Coords[0:Nnodes,1]
    Y = Coords[0:Nnodes,2]
    Z = Coords[0:Nnodes,3]    

    # Reshape all X,Y,Z arrays

    X = X.reshape(Nnodes,1)
    Y = Y.reshape(Nnodes,1)
    Z = Z.reshape(Nnodes,1)

    # Compute the FINAL discplacement field

    UX = X_CRD - X
    UY = Y_CRD - Y
    UZ = Z_CRD - Z

    # Update the Node Coordinates

    X = X_CRD
    Y = Y_CRD
    Z = Z_CRD

    # Read all Element information from the current mesh file 
    Element = pd.read_csv(Mesh_Name,header = None, sep = ',', skiprows = Nnodes + 4, nrows=Nelems)

    ELE = Element.to_numpy()

    # Check if correct number of elements are read from the *.nam file

    Element_Count = ELE[Nelems-1,0]

    if (Element_Count - Nelems > 0.00001):
        print("Did not read all elements from nam file")
        print("Read from *.nam ", Element_Count)
        print("Actual number of nodes ", Nelems)

    # Get the element connectivity for a tet (C3D4 mesh)

    Eleid = ELE[0:Nelems,0]
    E1 = ELE[0:Nelems,1]
    E2 = ELE[0:Nelems,2]
    E3 = ELE[0:Nelems,3]
    E4 = ELE[0:Nelems,4]

    # Uncomment the next two lines to check first and last read for elements

    #print(Eleid[0],E1[0],E2[0],E3[0],E4[0])
    #print(Eleid[-1],E1[-1],E2[-1],E3[-1],E4[-1])

    # Reshape all arrays to column major format

    Eleid= Eleid.reshape(Nelems,1)  
    E1= E1.reshape(Nelems,1)
    E2= E2.reshape(Nelems,1)
    E3= E3.reshape(Nelems,1)
    E4= E4.reshape(Nelems,1)

    # Round the new nodal coordinates to 6 digits of precicion
    for i in range(Nnodes):
        X[i,0] = round(X[i,0],6)
        Y[i,0] = round(Y[i,0],6)
        Z[i,0] = round(Z[i,0],6) 

    # Write the updated mesh file
    #Def_Mesh_Name = "Solid/deform_"+str(n)+".msh"
    Def_Mesh_Name = "Solid/deform.msh"

    file = open(Def_Mesh_Name, "w")
    file.write("  **This containes mesh information")
    file.write("\n  *NODE, NSET=NALL")
    #file.write("\n")
    for i in range(Nnodes):
        file.write("\n" '  ' + str(int(Nodeid[i,0])) +'\t , \t'+ str(X[i,0]) +'  \t, \t  '+ str(Y[i,0]) + '  , ' + str(Z[i,0]))
    file.write("\n")
    file.write("\n")
    #file.write("\n")
    #file.write("\n")
    file.write("*ELEMENT, TYPE=C3D4, ELSET=EALL")
    for i in range(Nelems):
        file.write("\n" + str(int(Eleid[i,0])) +' , '+ str(E1[i,0]) +' , '+ str(E2[i,0]) + ' , ' + str(E3[i,0])+ ' , ' +str(E4[i,0]))
    file.write("\n")
    file.close()
    
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------------------------------- EXTRACT STRESS FIELD -------------------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    

    Str = pd.read_csv('Stress.csv',header = None, sep = ',', skiprows = 3, nrows=Nnodes, engine='python')
    STRESS = Str.to_numpy()
    # Determine the number of surface nodes

    SXX =  STRESS[:,1]
    SYY =  STRESS[:,2]
    SZZ =  STRESS[:,3]
    SXY =  STRESS[:,4]
    SYZ =  STRESS[:,5]
    SZX =  STRESS[:,6]


    #print(STRESS.shape)
    #print(STRESS[:,1])
    #print(STRESS[:,2])
    #print(STRESS[:,3])

    SXX = SXX.reshape(Nnodes,1)
    SYY = SYY.reshape(Nnodes,1)
    SZZ = SZZ.reshape(Nnodes,1)
    SXY = SXY.reshape(Nnodes,1)
    SYZ = SYZ.reshape(Nnodes,1)
    SZX = SZX.reshape(Nnodes,1)

    # Compute von Mises stress for "general" state of stress
    Sigma = np.zeros(Nnodes)
    Sigma = Sigma.reshape(Nnodes,1)
    for i in range(Nnodes):
        Sigma[i,0] = math.sqrt((0.5*((SXX[i,0]-SYY[i,0])**2 + (SYY[i,0]-SZZ[i,0])**2 + (SZZ[i,0]-SXX[i,0])**2 + 3*(SXY[i,0]**2 + SYZ[i,0]**2 + SZX[i,0]**2))))

    #os.remove("Stress_read.pyc")
    

    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------------------------------- EXTRACT FORCE FIELD --------------------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

    # Deleted this code segment


    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#
    #----------------------------------------------------------------------------- WRITE VIZ FILES ------------------------------------------------------------------------------------------#
    #----------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------#

    TopOpt = False

    
     #Def_Mesh_Name = "Solid/deform_"+str(n)+".msh"
    file = open("elasticField_"+str(n)+".dat", "w")
    file.write("TITLE = \"Visualization of the Elastic field\"")
    file.write("\n")
    if TopOpt:
        file.write("VARIABLES = \"x\"\"y\"\"z\"\"Ux\"\"Uy\"\"Uz\"\"Fx\"\"Fy\"\"Fz\"\"Sigma\"\"Rho\"")
    else:
        file.write("VARIABLES = \"x\"\"y\"\"z\"\"Ux\"\"Uy\"\"Uz\"\"Sigma\"")

    file.write("\n")
    file.write("ZONE STRANDID=2, SOLUTIONTIME=2822, NODES=" + str(Nnodes) + ", ELEMENTS= " + str(Nelems) + ", DATAPACKING=POINT, ZONETYPE=FETETRAHEDRON")
    for i in range(Nnodes):
        if TopOpt:
            file.write("\n" +"{:.5e}" .format(X[i,0]) +'  \t  '+ "{:.5e}".format(Y[i,0]) + ' \t \t ' + "{:.5e}".format(Z[i,0]) + '\t'+ "{:.5e}".format(UX[i,0])+'\t'+ "{:.5e}".format(UY[i,0])+'\t'+ "{:.5e}".format(UZ[i,0]) + ' \t ' +"{:.5e}".format(FX[i,0]) + ' \t ' +"{:.5e}".format(FY[i,0]) + ' \t ' +"{:.5e}".format(FZ[i,0])+ ' \t ' +"{:.5e}".format(Sigma[i,0])+ ' \t ' +"{:.5e}".format(densityinterp_ave[i,0]))
        else:
            file.write("\n" +"{:.5e}" .format(X[i,0]) +'  \t  '+ "{:.5e}".format(Y[i,0]) + ' \t \t ' + "{:.5e}".format(Z[i,0]) + '\t'+ "{:.5e}".format(UX[i,0])+'\t'+ "{:.5e}".format(UY[i,0])+'\t'+ "{:.5e}".format(UZ[i,0]) +  ' \t ' +"{:.5e}".format(Sigma[i,0]))

    for i in range(Nelems):
        file.write("\n" + str(E1[i,0]) +' \t '+ str(E2[i,0]) + ' \t ' + str(E3[i,0])+ ' \t ' +str(E4[i,0]))
    file.write("\n")
    file.close()
    #print("Post processing Complete!")
    return Def_Mesh_Name    




