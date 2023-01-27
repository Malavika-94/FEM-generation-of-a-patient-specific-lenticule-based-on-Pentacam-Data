# -*- coding: utf-8 -*-
"""
SMILE_lenticule.py

Used to generate the patient specific lenticule mesh that can be used for Finite Element Analysis on ABAQUS.
The Geometry is based on elevation map from Pentcam with fintie element implementation on ABAQUS
The boundary conditions on the FE model are as illustrated in the figure. 

To use this script, you will need a csv file containing patient id & surgical parameters and the respective 
pentacam elevation file in csv format named as patient id



Created on Mon Aug  2 14:25:52 2021


"""
import numpy as np
import math
import gmsh
import sys
import matplotlib.tri as mtri
import trimesh
import os




# If your pentacam file is seperated with ; use this function
def read_pentacam_old(patient_id):
    # reads the pentacam csv file and return the point clouds describing  
    # find the correct pentacam file
    
    foldername = 'PentacamData/'
    filenames = next(os.walk(foldername), (None, None, []))[2]
    filename = [file for file in filenames if patient_id in file]
    # it is possible that the pentacam file is not available for the patient
    # if this is the case, just use another one, but let the user know
    if not filename: 
        print('Caution! Pentacam is not available. Anotherone is used!')
        filename = foldername + filenames[0]
    else:
        filename = foldername + filename[0] # TODO: multiple pentacams for one patient possible. at the moment first one is used
    
    
    # get the front and the back surfaces of the patient's cornea # Was Tab changed to ;
    separator = ";"
    f = open(filename, "rb")  # required on some files to read the thickness at the apex (solves an encoding issue)
    front = np.genfromtxt(f, delimiter=separator, skip_header=3 * 142, max_rows=142)
    f.seek(0)
    back_to_front = np.genfromtxt(f, delimiter=separator, skip_header=5 * 142, max_rows=142)
    f.close()

    # extract first column and row
    x = front[[0], 1:].ravel()
    x = x[:-1]  # remove last column (it's nan)
    y = np.transpose(front[1:, [0]]).ravel()

    # delete selected row & columns (describe x and y axis)
    front = front[1:, 1:-1]
    back_to_front = back_to_front[1:, 1:-1]
    back = back_to_front  # back_to_front is actually the posterior surface !!

    # coordinate of all the points
    xx, yy = np.meshgrid(x, y)

    # convert to vectors of np_points * 3 (x,y,z)
    points_front = np.stack([xx.reshape(141 * 141), yy.reshape(141 * 141), front.reshape(141 * 141)]).T
    points_back = np.stack([xx.reshape(141 * 141), yy.reshape(141 * 141), back.reshape(141 * 141)]).T

    # removes rows that include nan
    points_front = points_front[~np.isnan(points_front).any(axis=1)]
    points_back = points_back[~np.isnan(points_back).any(axis=1)]
    
    return (points_front, points_back)

# if your pentacam file is seperated with '-1' use this function 
def read_pentacam(patient_id):
 
    foldername = 'PentacamData/'
    filenames = next(os.walk(foldername), (None, None, []))[2]
    filename = [file for file in filenames if patient_id in file]
#    filename = 'C:/Users/malav/Universitaet Bern/Büchler, Philippe (ARTORG) - SNF - Indo-Swiss/SMILE/SMILE_WorkingDir/Generate_SMILE/PentacamData/Wrong/MP_OS_19032022_145022_ELE.csv'   
    if not filename: 
      print('Caution! Pentacam is not available. Anotherone is used!')
      filename = foldername + filenames[0]
    else:
      filename = foldername + filename[0] #multiple pentacams for one patient possible, at the moment first one is used
    # reads the pentacam csv file and return the point clouds describing
    # the front and the back surfaces of the patient's cornea

    # reading issues come from encoding, currently solved by saving file in utf8.csv and mentioning encoding in reader- issue for some cases
    front = np.genfromtxt(filename, encoding="utf8", skip_header=9, max_rows=141)
    back = np.genfromtxt(filename, encoding="utf8", skip_header=154, max_rows=141)

    front[front == -1] = 'nan'
    front = front[:,:-1]
    front[front == 0] = 'nan'
    
    back[back == -1] = 'nan'    
    back = back[:,:-1]

    # creates a 1-D array of the x -coordinates of the points by using np.linspace() with a range from -7 to 7 and 141 points
    # extract first column and row
    x= np.linspace(-7,7,141)
    y = np.transpose(x).ravel()
    # coordinate of all the points
    xx, yy = np.meshgrid(y, y)      

    # convert to vectors of np_points * 3 (x,y,z), convert to um 

    points_front = np.stack([xx.reshape(141 * 141), yy.reshape(141 * 141), front.reshape(141 * 141)/ 1000]).T
    points_back = np.stack([xx.reshape(141 * 141), yy.reshape(141 * 141), back.reshape(141 * 141)/ 1000]).T

    # removes rows that include nan
    points_front = points_front[~np.isnan(points_front).any(axis=1)]
    points_back = points_back[~np.isnan(points_back).any(axis=1)]
    
 # returns front & back surface point clouds
    return (points_front, points_back)

def ellipsoidFit(X):
    # fit an ellipsoid on pointcloud X
    x = X[:, 0]
    y = X[:, 1]
    z = X[:, 2]
    D = np.array([x * x + y * y - 2 * z * z,
                  x * x + z * z - 2 * y * y,
                  2 * x * y,
                  2 * x * z,
                  2 * y * z,
                  2 * x,
                  2 * y,
                  2 * z,
                  1 - 0 * x])
    d2 = np.array(x * x + y * y + z * z).T  # rhs for LLSQ
    u = np.linalg.solve(D.dot(D.T), D.dot(d2))
    a = np.array([u[0] + 1 * u[1] - 1])
    b = np.array([u[0] - 2 * u[1] - 1])
    c = np.array([u[1] - 2 * u[0] - 1])
    v = np.concatenate([a, b, c, u[2:]], axis=0).flatten()
    A = np.array([[v[0], v[3], v[4], v[6]],
                  [v[3], v[1], v[5], v[7]],
                  [v[4], v[5], v[2], v[8]],
                  [v[6], v[7], v[8], v[9]]])

    center = np.linalg.solve(- A[:3, :3], v[6:9])

    translation_matrix = np.eye(4)
    translation_matrix[3, :3] = center.T

    R = translation_matrix.dot(A).dot(translation_matrix.T)

    evals, evecs = np.linalg.eig(R[:3, :3] / -R[3, 3])
    evecs = evecs.T

    radii = np.sqrt(1. / np.abs(evals))
    radii *= np.sign(evals)

    return center, evecs, radii, v

# Edit your patients epithelium thickness and surgical cap thickness here in epi thickness and depth --
def remove_epithelium(points_front, points_back):
    epi_thickness   = 50e-3 # epithelium thickness (mm)
    depth           = 0.13 - epi_thickness # of pocket or SMILE (in mm)  ## Should be 130 or 115
    #get an unstructured triangular grid
    tri = mtri.Triangulation(points_front[:, 0], points_front[:, 1])
    # create 3D mesh to get the normals of each face
    mesh = trimesh.Trimesh(vertices=points_front, faces=tri.triangles, process=False)
    # INFO: surface is opened to the top
    # remove epithelium -> add thickness to front surface
    points_front = points_front + epi_thickness * mesh.vertex_normals
    # move points back to 0 (front and back-surface)
    points_front = points_front - np.array([0, 0, epi_thickness])
    points_back =  points_back - np.array([0, 0, epi_thickness])  
            
    # middle points calculated as the front surface + depth (not so important for creation of just the SMILE lenticule)
    points_middle = points_front + np.array([0., 0., depth])   
    return points_middle

# surgical ablation profile calculation considering sphere and cylinder
def mrochen(points_middle, surgical_zone,ablation_zone,sphere,cylinder,angle,curvature):
    
    # variables for plotting
    x_plot = []
    y_plot = []
    
    # create grid
    nbPoints = 21
    x = np.linspace(-ablation_zone * 1.05 / 2, ablation_zone * 1.05 / 2, nbPoints)
    y = np.linspace(-ablation_zone * 1.05 / 2, ablation_zone * 1.05 / 2, nbPoints)
    
    # get the distance to the middle ellipsoid
    center, rot, axes, v = ellipsoidFit(points_middle)
    A = np.diag([1 / axes[0] ** 2, 1 / axes[1] ** 2, 1 / axes[2] ** 2])
    Q = np.dot(rot.T, np.dot(A, rot))
    distanceToEllipsoid = np.zeros((len(x), len(y)))
    for ii in range(len(x)):
        for jj in range(len(y)):
            origin = np.array([x[ii], y[jj], 0])  # origin of the line
            direction = np.array([0, 0, 1])  # direction of the line
            p = origin - center
            aa = np.dot(direction.T, np.dot(Q, direction))
            bb = 2 * np.dot(direction.T, np.dot(Q, p))
            cc = np.dot(p.T, np.dot(Q, p)) - 1
    
            delta = bb ** 2 - 4 * aa * cc
            if delta >= 0:
                lambda1 = (-bb + np.sqrt(delta)) / (2 * aa)
                lambda2 = (-bb - np.sqrt(delta)) / (2 * aa)
                if np.abs(lambda1) < np.abs(lambda2):
                    distanceToEllipsoid[ii, jj] = origin[2] + lambda1 * direction[2]
                else:
                    distanceToEllipsoid[ii, jj] = origin[2] + lambda2 * direction[2]
                       
    # calculate ablation depth according to formula of mrochen
    sphere = abs(sphere)
    cylinder = abs(cylinder)
    R = surgical_zone/2/1000
    alpha = (90 + angle ) * math.pi/180
    ablation = np.zeros((len(x),len(y)))
    for ii in range(len(x)):
        for jj in range(len(y)):
            r = np.linalg.norm(np.array([x[ii], y[jj]])) # variable radius
            # now: normalize radius
            rho = r/1000 /R  
            theta = math.atan2(y[jj], x[ii])
            # formula for the ablation
            ablation[ii,jj] =  R ** 2 / (2 * (1.337 - 1)) * (sphere * (1 - rho ** 2) + (cylinder / 2) * (
                            2 - rho ** 2 - rho ** 2 * math.sin(2 * alpha) * math.sin(2 * theta) - rho ** 2 * math.cos(
                        2 * alpha) * math.cos(2 * theta)))
            # to visualize ablatoin profile 
            if ii == nbPoints//2:
                x_plot.append(rho*R*1000)
                y_plot.append(ablation[ii,jj]*1e6)
                
    ablation = ablation * 1000 - 0.e-3  # convert from m to mm and add a small correction to ensure suitable projections            
                
    
    # define surface points (anterior and posterior)        
    anteriorSurfacePoints = np.zeros((len(x),len(y)))
    posteriorSurfacePoints = np.zeros((len(x),len(y)))

    for ii in range(len(x)):
        for jj in range(len(y)):
            anteriorSurfacePoints[ii,jj] = -distanceToEllipsoid[ii,jj]
    
    # anterior surface is not at 0 -> adjust for offset
    offset = np.max(anteriorSurfacePoints)
    for ii in range(len(x)):
        for jj in range(len(y)):
            anteriorSurfacePoints[ii,jj] = anteriorSurfacePoints[ii,jj] - offset
            posteriorSurfacePoints[ii,jj] = -distanceToEllipsoid[ii,jj] - ablation[ii,jj] - offset    
    # use gmsh to define points and surfaces  
    BSplinePoints_anterior = []
    for ii in range(len(x)):
        for jj in range(len(y)):
            BSplinePoints_anterior.append(gmsh.model.occ.addPoint(x[ii], y[jj], anteriorSurfacePoints[ii,jj]))
    
    BSplinePoints_posterior = []
    for ii in range(len(x)):
        for jj in range(len(y)):
            BSplinePoints_posterior.append(gmsh.model.occ.addPoint(x[ii], y[jj], posteriorSurfacePoints[ii,jj]))

    ablationBSpline_anterior = gmsh.model.occ.addBSplineSurface(BSplinePoints_anterior, nbPoints)
    ablationBSpline_posterior = gmsh.model.occ.addBSplineSurface(BSplinePoints_posterior, nbPoints)
 
    #define the volumes. They are extruded first and then, one is subtracted from the other
    ablationBSplineVolume_anterior = gmsh.model.occ.extrude([(2,ablationBSpline_anterior)],0,0,11)
    ablationBSplineVolume_posterior = gmsh.model.occ.extrude([(2,ablationBSpline_posterior)],0,0,10)
    ablationBSplineVolume_anterior = np.asarray(ablationBSplineVolume_anterior)
    ablationBSplineVolume_posterior = np.asarray(ablationBSplineVolume_posterior)
    splineVolume_anterior = ablationBSplineVolume_anterior[np.where(ablationBSplineVolume_anterior[:,0] == 3), 1][0][0]
    splineVolume_posterior = ablationBSplineVolume_posterior[np.where(ablationBSplineVolume_posterior[:,0] == 3), 1][0][0]
    # # delete gmsh points used to define the BSplineSurface
    gmsh.model.occ.remove([(0, p) for p in BSplinePoints_anterior])     
    gmsh.model.occ.remove([(0, p) for p in BSplinePoints_posterior])
    # remove one volume from the other to get the ablation volume
    
    gmsh.model.occ.cut([(3,splineVolume_posterior)], [(3, splineVolume_anterior)])  
    # before we cut the lenticule into the correct shape (diameter of cylinder) -> add thickness 
    # this step is done now, to have simpler surfaces in gmsh
    add_lenticule_thickness(sphere)
    # add cylinder with the diameter equals to the ablation zone
    cyl = gmsh.model.occ.addCylinder(0, 0, 0, 0, 0, -4, ablation_zone/2)
    lenticule = gmsh.model.getEntities(3)
    
    gmsh.model.occ.intersect(lenticule, [(3,cyl)])
    gmsh.model.occ.synchronize()

# If your surgical correction has a plane parallel layer, add this here, else set to 0
def add_lenticule_thickness(correction):
    # thickness depends on correction
    if correction < 3:
        thickness = 40e-3
    elif correction < 6:
        thickness = 30e-3
    else:
        thickness = 20e-3
    gmsh.model.occ.synchronize()
    
    # find the id of the posterior surface  
    surfaces = []
    center_of_mass = []
    for s in gmsh.model.getEntities(2):
        if gmsh.model.getType(s[0],s[1]) == 'BSpline surface':
            surfaces.append(s[1])
            center_of_mass.append(gmsh.model.occ.getCenterOfMass(s[0], s[1]))
    if len(surfaces) != 2:
        print('Warning: number of surfaces is incorrect')
    else:
        if center_of_mass[0][2] < center_of_mass[1][2]:
            id_post = surfaces[0]
        else:
            id_post = surfaces[1]
            
    # copy posterior surface   
    copy_surf = gmsh.model.occ.copy([(2,id_post)])
    # extrude copied surface
    gmsh.model.occ.extrude(copy_surf, 0, 0, -thickness)
    # fuse new volume to existing lenticule
    gmsh.model.occ.fuse([(3,2)],[(3,3)])
    gmsh.model.occ.synchronize()

    
def holes_in_lenticule(distance):
    diameter = 0.254 #from BioRakes case, adapt to your experiment
    radius = diameter/2
    spacing = 0.7 #from BioRakes case, adapt to your experiment
    gmsh.model.occ.synchronize()
    #fixed side
    gmsh.model.occ.addCylinder(-distance/2, 0, 0, 0, 0, -1, radius)
    gmsh.model.occ.cut([(3,1)], [(3,2)])
    gmsh.model.occ.addCylinder(-distance/2, -(spacing), 0, 0, 0, -1, radius)
    gmsh.model.occ.cut([(3,1)], [(3,2)])
    gmsh.model.occ.addCylinder(-distance/2, spacing, 0, 0, 0, -1, radius)
    gmsh.model.occ.cut([(3,1)], [(3,2)])
    #moved side
    gmsh.model.occ.addCylinder(distance/2, 0, 0, 0, 0, -1, radius)
    gmsh.model.occ.cut([(3,1)],[(3,2)])
    gmsh.model.occ.addCylinder(distance/2, -(spacing), 0, 0, 0, -1, radius)
    gmsh.model.occ.cut([(3,1)],[(3,2)])
    gmsh.model.occ.addCylinder(distance/2, spacing, 0, 0, 0, -1, radius)
    gmsh.model.occ.cut([(3,1)],[(3,2)])
    gmsh.model.occ.synchronize()
     
 
#surfaces have to be identified to assign BC's. 
#This is done via the position and type of surface.   
def identify_surfaces():

    # get all the volumes of the model
    gmsh.model.occ.synchronize()
    allVolumes = gmsh.model.occ.getEntities(3)
    # get boundaries of the volumes
    all_surfaces = gmsh.model.getBoundary(allVolumes)
    id_periphery_surface = []
    ext_surfaces = []
    cyl_surfaces = []
    other_surfaces = []

    # sort the surfaces, pos_surfaces becoz of negative tag unrecognition on gmsh 4.10
    pos_all_surfaces = [(abs(s[0]), abs(s[1])) for s in all_surfaces]
    
    for s in pos_all_surfaces:
        if gmsh.model.getType(s[0],s[1]) == 'BSpline surface':
            ext_surfaces.append(s[1])
        elif gmsh.model.getType(s[0], s[1]) == 'Cylinder':
            cyl_surfaces.append(s[1])
        else:
            other_surfaces.append(s[1])
            # print('Other surfaces detected')

    id_cylfix_surface = []
    id_cylencastre_surface = []
    id_cylmove_surface = []
    for surf in cyl_surfaces:
        # get the coordinates of the Center of Mass for each surfaces
        coord = gmsh.model.occ.getCenterOfMass(2,surf)
        # find surface with CoM in the middle -> periphery surface
        if abs(coord[0]) < 0.01:
            id_periphery_surface.append(surf)
        # find surfaces for the fixed holes (negative x-axis at ~-2.125)
        elif -2.35 < coord[0] < -2.10 and abs(coord[1]) < 1:
            if abs(coord[1]) > 0.01: # just a small number to get cylinder in the middle
                id_cylfix_surface.append(surf)
            else:
                id_cylencastre_surface.append(surf)
        # all other cylinder-surfaces are for the moved holes
        elif 2.1 < coord[0] < 2.35 and abs(coord[1]) < 1:
            id_cylmove_surface.append(surf)
 
    # find normal of the external surfaces
    orient = []
    for surf in ext_surfaces:
        param = gmsh.model.getParametrizationBounds(2, surf)
        normal = gmsh.model.getNormal(surf, (param[0] + param[1])/2)
        orient.append(normal.dot([0,0,1]))
           
    orient = np.asarray(orient)
    id_ant_surface = []
    id_post_surface = []
    
    # depending on the orientation of the normal we can define the anterior and posterior surface
    for ii in range(len(ext_surfaces)):
        if ( orient[ii] > 0):
            id_ant_surface.append(ext_surfaces[ii])
        else:
            id_post_surface.append(ext_surfaces[ii])
         
    return (id_periphery_surface, id_ant_surface, id_post_surface, id_cylencastre_surface, id_cylfix_surface, id_cylmove_surface)


# Creates Gmsh mesh, uncoomment the fltk.run line to visualise on gmsh
def mesh_model():
    gmsh.model.occ.synchronize()
    gmsh.model.removePhysicalGroups()
    
# edit below lengthMin and Max parameters to change mesh size based on your needs
    gmsh.option.setNumber("Mesh.CharacteristicLengthMin", 0.08)
    gmsh.option.setNumber("Mesh.CharacteristicLengthMax", 0.12)
 
    gmsh.option.setNumber("Mesh.MeshSizeFromCurvature", 2) # was active
      
    lcar1 = 0.28
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), lcar1)
    
#set elements to second order or first as needed
    gmsh.option.setNumber("Mesh.ElementOrder", 2)
    gmsh.option.setNumber("Mesh.SecondOrderLinear", 1)
#    gmsh.fltk.run()    
    gmsh.model.mesh.generate(3)

#Writes Abaqus inp file -- edit BC's and loading here
def write_Abaqus(filename, surface_ids, disp):
    # define surface id's
    id_periphery_surface, id_ant_surface, id_post_surface, id_cylencastre_surface, id_cylfix_surface, id_cylmove_surface = surface_ids

    f = open(filename, "w")
    # write header
    f.write('*Heading\n')
    f.write('**Smile lenticule - Mesh generated with gmsh\n')    
    
    gmsh.model.mesh.renumberNodes()
    gmsh.model.mesh.renumberElements()
    
    gmsh.model.occ.synchronize()
   
    allVolumes = gmsh.model.occ.getEntities(3)
    if len(allVolumes) != 1:
        print('There should only be one volume!!!\n')
    id_volume = allVolumes[0][1]
    
    physicalGroupAllSolids = gmsh.model.addPhysicalGroup(3, [id_volume])
    nTags, nCoord = gmsh.model.mesh.getNodesForPhysicalGroup(3, physicalGroupAllSolids)
    nCoord = nCoord.reshape(-1, 3)
    nTagsCood = np.column_stack((nTags, nCoord))
    f.write('*Node\n')
    nTagsCood = nTagsCood[nTagsCood[:, 0].argsort()]  # sort by increasing node number
    np.savetxt(f, np.asarray(nTagsCood), fmt=['%i', '%f', '%f', '%f'], delimiter=',\t')
    
    # write elements
    elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(3, id_volume)
    if elementTypes[0] == 4:
        nodeTags = nodeTags[0].reshape(-1, 4)  # linear tetrahedron with 4 nodes
        f.write('*Element, type=C3D4H, Elset= lenticule \n')
    elif elementTypes[0] == 11:
        nodeTags = nodeTags[0].reshape(-1, 10)  # quadratic tetrahedron with 10 nodes
        nodeTags[:, [8, 9]] = nodeTags[:, [9, 8]]  # invert postition of nodes 9 and 10 (gmsh 8 and 9)
        f.write('*Element, type=C3D10H, Elset= lenticule \n')
    
    eTagsNodes = np.column_stack((elementTags[0], nodeTags))
    eTagsNodes = eTagsNodes[eTagsNodes[:, 0].argsort()]  # sort by increasing element number
    np.savetxt(f, np.asarray(eTagsNodes), fmt='%i', delimiter=',\t')

    # write elsets and nsets
    
    # write an elset with all the elements
    f.write('*Elset, elset=All\n')
    f.write('lenticule\n')
    
    
    # write nset anterior surface
    f.write('*Nset, nset=anterior_nset\n')
    physicalGroupSurfaceAnterior = gmsh.model.addPhysicalGroup(2, id_ant_surface)
    anteriorTags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(2, physicalGroupSurfaceAnterior)
    anteriorTags = np.sort(anteriorTags)  # sort by increasing node number
    np.savetxt(f, anteriorTags.T, fmt='%i', delimiter=',\t')
    
    # write nset posterior surface
    f.write('*Nset, nset=posterior_nset\n')
    physicalGroupSurfacePosterior = gmsh.model.addPhysicalGroup(2, id_post_surface)
    posteriorTags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(2, physicalGroupSurfacePosterior)
    posteriorTags = np.sort(posteriorTags)  # sort by increasing node number
    np.savetxt(f, posteriorTags.T, fmt='%i', delimiter=',\t')
   
    # write nset for encastred holes
    f.write('*Nset, nset=cylinder_encastre_nset\n')
    physicalGroupSurfaceCylEnc = gmsh.model.addPhysicalGroup(2, id_cylencastre_surface)
    periphTags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(2, physicalGroupSurfaceCylEnc)
    periphTags = np.sort(periphTags)  # sort by increasing node number
    np.savetxt(f, periphTags.T, fmt='%i', delimiter=',\t')    
    
    # write nset for fixed holes
    f.write('*Nset, nset=cylinder_fix_nset\n')
    physicalGroupSurfaceCylFix = gmsh.model.addPhysicalGroup(2, id_cylfix_surface)
    periphTags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(2, physicalGroupSurfaceCylFix)
    periphTags = np.sort(periphTags)  # sort by increasing node number
    np.savetxt(f, periphTags.T, fmt='%i', delimiter=',\t')
    
    # write nset for moved holes
    f.write('*Nset, nset=cylinder_move_nset\n')
    physicalGroupSurfaceCylMove = gmsh.model.addPhysicalGroup(2, id_cylmove_surface)
    periphTags, _ = gmsh.model.mesh.getNodesForPhysicalGroup(2, physicalGroupSurfaceCylMove)
    periphTags = np.sort(periphTags)  # sort by increasing node number
    np.savetxt(f, periphTags.T, fmt='%i', delimiter=',\t')

    
    elementTypes, elementTags, nodeTags = gmsh.model.mesh.getElements(3, 1)
    
    # construct a list containing all the faces of all the elements
    if elementTypes[0] == 4:
        tt = nodeTags[0].reshape(-1, 4)
    elif elementTypes[0] == 11:
        tt = nodeTags[0].reshape(-1, 10)
        
    face1 = np.column_stack((tt[:, 0], tt[:, 1], tt[:, 2]))
    face2 = np.column_stack((tt[:, 0], tt[:, 1], tt[:, 3]))
    face3 = np.column_stack((tt[:, 1], tt[:, 2], tt[:, 3]))
    face4 = np.column_stack((tt[:, 0], tt[:, 2], tt[:, 3]))
    abaqusFaces = np.vstack((face1, face2, face3, face4))
    
    physicalGroupSurface = gmsh.model.addPhysicalGroup(2, id_cylmove_surface)
        
    nodesSurface, _ = gmsh.model.mesh.getNodesForPhysicalGroup(2, physicalGroupSurface)
    
    node_1 = np.in1d(abaqusFaces[:, 0], nodesSurface)
    node_2 = np.in1d(abaqusFaces[:, 1], nodesSurface)
    node_3 = np.in1d(abaqusFaces[:, 2], nodesSurface)
    faces_surface = node_1 & node_2 & node_3
    index_faces = np.where(faces_surface)[0]
    
    # create the list of the ID of all the element faces
    all_elements_face_ID = np.column_stack(
        (np.tile(elementTags[0], 4), np.repeat(["S1", "S2", "S3", "S4"], len(elementTags[0]))))
    
    # select in the list the element faces with all three nodes on the surface
    elementFaces = all_elements_face_ID[index_faces]
    

    # write posterior surface
    f.write('*Surface, name=cylinder_move_surface\n')
    elementFaces = elementFaces[elementFaces[:, 0].argsort()]  # sort by increasing element number
    np.savetxt(f, np.asarray(elementFaces), fmt='%s', delimiter=',\t')
     
    # Constraint
      
    # get middle of moved surface
    CoM = gmsh.model.occ.getCenterOfMass(2,id_cylmove_surface[1])
    
    f.write('*Node\n')
    referencePoint=np.array([len(nTags)+1, CoM[0], 0, CoM[2]])
    referencePoint= referencePoint.reshape(1,4)
    np.savetxt(f,referencePoint, fmt=['%i', '%f', '%f', '%f'], delimiter=',\t')
    f.write('*Nset, nset=ReferenceSet\n')
    np.savetxt(f,np.array([len(nTags)+1]), fmt='%i')
    f.write('** Constraint: ConstraintReference\n')
    f.write('*Coupling, constraint name=ConstraintReference, ref node=ReferenceSet, surface=cylinder_move_surface\n')
    f.write('*Kinematic\n')
    # Couple just x-direction
    f.write('1, 1\n')    
    
    #add orientation

    f.write('*Orientation, name=Ori-1\n')
    f.write('1, 0,           0., 0, 1,           0.\n')
    f.write('3, 0.\n')
    f.write('1,0,0\n') #local direction of fibres
    f.write('0,1,0\n') # local direction of fibres
    
    # Section
    f.write('** SECTION\n')
    f.write('*Solid Section, elset=All, orientation=Ori-1, material=Material-1\n')
    f.write(',\n')
    
    # Material
    f.write('** MATERIAL\n')
    f.write('*Material, name=Material-1\n')
#    f.write('*Include, input=coefficients.inp\n') # This file contains include user material and the parameters
 
   # Step --  Two steps are written here - first one can be used for setting your Zero lenghth value / prestretch parameter
   # 2nd step for actual loading 
    
    f.write('********************\n')
    f.write('** STEP: Step1\n')
    f.write('*Step, name=Step-step1, nlgeom=YES, inc=1000\n')
    f.write('*Static\n')
    f.write('0.01, 1, 1E-04, 1\n')
    
#Adapt this section to account for your experimental setting       
    # Just encastre the middle hole. Allow y-z-movement for the other two
    f.write('** BOUNDARY CONDITIONS\n')
    f.write('*Boundary\n')
    f.write('cylinder_fix_nset, 1,1\n')       
    f.write('*Boundary\n')
    f.write('cylinder_encastre_nset, ENCASTRE\n')
    
    # Displacement     
    f.write('*Boundary\n')
#    f.write('ReferenceSet, 1, 1, ValueforPrestretch\n')
    f.write('ReferenceSet, 1, 1\n')
    f.write('*Boundary\n')
    f.write('ReferenceSet, 2, 2\n')
    f.write('*Boundary\n')
    f.write('ReferenceSet, 3, 3\n')
    f.write('*Boundary\n')
    f.write('ReferenceSet, 4, 4\n')
    f.write('*Boundary\n')
    f.write('ReferenceSet, 5, 5\n')
    f.write('*Boundary\n')
    f.write('ReferenceSet, 6, 6\n')   
    #output    
    f.write('**OUTPUT REQUEST\n')
    f.write('*Output, history\n')
    f.write('*Node Output, nset=ReferenceSet\n')
    f.write('RF1, U1,\n')  
    # End step
    f.write('*End Step\n')
    
    #Start second step
    f.write('********************\n')
    f.write('** STEP: Step2\n')
    f.write('*Step, name=Step-step2, nlgeom=YES\n')
    f.write('*Static\n')
    f.write('0.01, 1, 1E-04, 1\n')
          
    # BC
    f.write('** BOUNDARY CONDITIONS\n')
    f.write('*Boundary\n')
    f.write('ReferenceSet, 1, 1, %s\n' %disp)     
    #output    
    f.write('**OUTPUT REQUEST\n')
    f.write('*Output, history\n')
    f.write('*Node Output, nset=ReferenceSet\n')
    f.write('RF1, U1,\n')  
    # End step
    f.write('*End Step')
   
    f.close()
    
def generate_model(patient_information):
    
    # get parameters from information list
    patient_id, surgical_zone, ablation_zone, sphere, cylinder, angle, curvature, holes_dist, disp = patient_information
    
    # initialize gmsh    
    gmsh.initialize()
    # gmsh.option.setNumber("General.Terminal", 1)
    # add gmsh model
    gmsh.model.add("lens")
     # get pentacam data
    points_front, points_back = read_pentacam(patient_id) 
#    points_front, points_back = read_pentacam(file) 
        # remove the epithelium thickness and moveback the points to have the apex (after removal) at the origin
    points_middle = remove_epithelium(points_front, points_back)
    
    # calculate and model the ablation profile
    mrochen(points_middle, surgical_zone, ablation_zone, sphere, cylinder, angle, curvature)  

    holes_in_lenticule(holes_dist) # Get holes into the lenticule (4.25mm apart)
    # identify surfaces of complete model
    id_periphery_surface, id_ant_surface, id_post_surface, id_cylencastre_surface, id_cylfix_surface, id_cylmove_surface = identify_surfaces()
    
    surface_ids = identify_surfaces()
    # mesh model
    mesh_model()
    
    gmsh.model.occ.synchronize() 
#    gmsh.fltk.run()
#    gmsh.finalize()
    
    # write abaqus inp file in folder called PatientInput 
    
    filename = os.pardir + "/PatientInput/" + patient_id + '.inp'
    write_Abaqus(filename, surface_ids, disp)
      
    gmsh.finalize()

def get_inp_files(patient_information):
    # get patient id from list
    patient_id = patient_information[0]
    # line showing the progress of the file generation
    print(patient_id + ' --> get .inp files...')
    generate_model(patient_information)
   
    
def condense_data(data):
    # used to get rid of empty spaces in data
    data_cond = []
    for i in data:
        if i:
            data_cond.append(i.decode("utf-8"))
            
    return data_cond

#get surgical parameters from csv file, edit file name containing patient file if needed
def get_patient_parameters(patient_list):
    # dir where patient data is stored. Or file with relevant data.
    filename = 'patient_data.csv'
    
    #get the first letters of the names & eye_id to create patient ID (John Doe's right eye -> JD_OD)
#
    f=open(filename,'rb')
    lastname_id = np.genfromtxt( f, delimiter=',', usecols=(2), dtype='S1', skip_header=2)
    f.seek(0)
    firstname_id = np.genfromtxt( f, delimiter=',', usecols=(3), dtype='S1', skip_header=2)
    f.seek(0)
    eye_id = np.genfromtxt( f, delimiter=',', usecols=(5), dtype='S3',skip_header=2)
    f.seek(0)
    # get relevant information
    # OZ diameter | r of curvature | corr. sphere | corr. cylinder | corr. axis | Holes_dist

    patient_params = np.genfromtxt( f, delimiter=',', usecols=(8,13,14,15,16,17,18), skip_header=2, skip_footer=0)
    f.close()
    #condense data (get rid of empty spaces)
    lastname_id = condense_data(lastname_id)
    firstname_id = condense_data(firstname_id)
    eye_id = condense_data(eye_id)
      
    # duplicate the name_ids (initials needed for both eyes)
    lastname_id = np.repeat(lastname_id, repeats=2, axis=0)
    firstname_id = np.repeat(firstname_id, repeats=2, axis=0)
    
    # combine initials and eye_id into one string
    patient_id = []
    # get patient_id
    for i in range(0,len(lastname_id)):
        patient_id.append(lastname_id[i] + firstname_id[i] + '_' + eye_id[i])
        
    #create a dictionary with all patient_ids
    patient_parameters = dict(zip(patient_id, patient_params))
    
    #reduce dictionary to patients from which the parameters are wanted (in patient_list)
    if not patient_list:
        #if list is empty, get all the patients
        return patient_parameters
    else:
        temp_parameters = patient_parameters.copy()
        for i in temp_parameters:
            if i not in patient_list:
                patient_parameters.pop(i)
                
        return patient_parameters


def main():
   
    # get patient parameters. List of wanted patients as input. If a model of all patients is needed, use empty list []- Edit patient id here 
    patient_parameters = get_patient_parameters(["PATIENT1"])
    
    # run a loop for all patients in the list to get patameters
    # In this study, parameters used are Surgical Zone, spherical correction, cylindrical correction, its axis (angle), the radius of curvature
    # the distance between hooks/clamps/ loading of experiment & the displacement to be imposed
    
    for i in patient_parameters:
        # extract relevant parameters, order given below
        # this step is not really necessary (parameters could be directly stored into a list), but good to see which parameter is which
        patient_id = i
        surgical_zone = patient_parameters[i][0]
        ablation_zone = surgical_zone
        sphere = patient_parameters[i][2]
        cylinder = patient_parameters[i][3]
        angle = patient_parameters[i][4]
        curvature = patient_parameters[i][1]
        holes_dist =patient_parameters[i][5]
        disp = patient_parameters[i][6]

        # save parameters into list
        patient_information = [patient_id, surgical_zone, ablation_zone, sphere, cylinder, angle, curvature, holes_dist, disp]
        # pass patient information to get the input files
        get_inp_files(patient_information)
  
if __name__ == "__main__":
    main()
