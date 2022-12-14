
def wrtInput(Mesh_Name, Dynamic, E, nu, rho):

    description = 'Input deck for elastic simulation'   # Add some details for inp file header, keep one line

    filename={}
    if Dynamic == 1:
        filename['ccxdeck']="elastic.inp"    # ccx inp file
        filename['meshfile']='Solid/'+str(Mesh_Name)    # ccx mesh file
        filename['fixedNodes']='Solid/bottomNodes.nam'
        filename['surfaceNodes']='Solid/surfaceNodes.nam'
    if Dynamic == 0:
        filename['ccxdeck']="elastic.inp"    # ccx inp file
        filename['meshfile']='Solid'+str(Mesh_Name)    # ccx mesh file
        filename['fixedNodes']='Solid/bottomNodes.nam'
        filename['surfaceNodes']='Solid/surfaceNodes.nam'
        filename['surfaceForces']='Solid/surfaceForces.nam' 

    element = {}
    element['type']='C3D4'     # C3D4 = tetrahedral, C3D8 = hexahedral, CPS3 = tri, CPS4 = quad
    element['E']=str(E)     # Young's Modulus
    element['poisson']=str(nu)   # Poisson ratio
    element['density']=str(rho)   # Density (optional for cload static case)


    # Note: By default, inp file assumes tags[0] =  fixed node set, tags[1] = load node sets. This order is assumed inside inp file 
    tags=['Nfix1', 'Nsurface']

    load={}     
    load['x'] = '0.0'     
    load['y'] = '0.0'
    load['z'] = '0.0'

    set={}
    set['nodes'] = 'NALL'   #Set for all nodes
    set['elements'] = 'EALL'    #Set for all elements

    FLAGS = {}
    FLAGS['WRITE_INP'] = True

    FLAGS['TOPOPT'] = False # For topology optimization, additional flags are added

    if FLAGS['WRITE_INP']:
        with open(filename['ccxdeck'],"w") as file:
            file.write("**"+description)  
            file.write("\n*INCLUDE, INPUT="+filename['meshfile'])
            file.write("\n*INCLUDE, INPUT="+filename['fixedNodes'])
            file.write("\n*INCLUDE, INPUT="+filename['surfaceNodes'])
            

        #for tag_index, tag_name in enumerate(tags):
        #    tagfilename = tag_name + ".nam"
        #    file.write("\n*INCLUDE, INPUT=" + tagfilename)

            file.write("\n\n**Add Boundary file")        
            file.write("\n*BOUNDARY") 
            file.write("\n"+tags[0]+",1,3")

            file.write("\n\n**Constrain axis if required")        
            file.write("\n*BOUNDARY") 
            file.write("\n"+set['nodes']+",3,3")    
            
            if Dynamic == 0:
            #if FLAGS['TOPOPT']:
                file.write("\n\n*NSET, NSET = DV")        
                file.write("\n1") 

                file.write("\n\n*DESIGNVARIABLES, TYPE = COORDINATE")        
                file.write("\nNNN") 

            file.write("\n\n**Add material named EL with elastic properties")
            file.write("\n*MATERIAL,NAME=EL")
            file.write("\n*ELASTIC")
            file.write("\n"+element['E']+","+element['poisson'])

            file.write("\n\n*SOLID SECTION, ELSET="+set['elements']+", MATERIAL=EL")
            file.write("\n*DENSITY")
            file.write("\n"+element['density'])
            
            if Dynamic == 0:
                file.write("\n\n*STEP")        
                file.write("\n*STATIC")
                file.write("\n*CLOAD")
                file.write("\n*INCLUDE, INPUT = Solid/surfaceForces.nam")
            else:
                file.write("\n\n*STEP,NLGEOM, INC = 1000000000000")
                file.write("\n*DYNAMIC,DIRECT")
                file.write("\n1,1")        
                file.write("\n*CLOAD")
                file.write("\n"+tags[1]+",1,"+ load['x'])         
                file.write("\n"+tags[1]+",2,"+ load['y'])
                file.write("\n"+tags[1]+",3,"+ load['z'])       

            file.write("\n\n*NODE FILE")        
            file.write("\nU")
            file.write("\n*EL FILE")        
            file.write("\nS")            
            file.write("\n*NODE PRINT, NSET="+set['nodes'])        
            file.write("\nU")
            file.write("\n*EL PRINT,ELSET="+set['elements'])        
            file.write("\nS")            
            file.write("\n*END STEP") 

            if Dynamic ==0:
            #if FLAGS['TOPOPT']:  
                file.write("\n\n*STEP")        
                file.write("\n*SENSITIVITY")
                file.write("\n*OBJECTIVE")        
                file.write("\nSTRAINENERGY")                      
                file.write("\n*END STEP")                 

    return 0

