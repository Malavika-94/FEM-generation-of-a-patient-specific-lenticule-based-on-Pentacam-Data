# FEM-generation-of-a-patient-specific-lenticule-based-on-Pentacam-Data
Finite element mesh generation of a patient-specific lenticule based on clinical Pentacam Data

# Finite element mesh generation of a patient-specific lenticule based on clinical Pentacam Data

Used to generate the patient-specific lenticule mesh from Pentacam Elevation maps that can be used for Finite Element Analysis on ABAQUS.
# About
This code is designed to generate a finite element mesh of human corneal lenticules obtained from CLEAR surgery patients using GMSH 4.0 and ABAQUS 2020. The purpose of this study is to improve our understanding of the biomechanical properties of young human corneal tissue, which is important for accurate refractive surgery procedures. The generated mesh can be used to simulate the mechanical behavior of the cornea using the Holzapfel Gasser Ogden (HGO) material model , ...
This code is especially relevant for researchers and practitioners in the field of ophthalmology, as it allows for the generation of patient-specific models of corneal lenticules. This can lead to a better understanding of the microstructure and mechanics of the cornea, which is crucial for the development of new surgical techniques and the improvement of existing ones. 
It is assumed that the user has the necessary software and knowledge to edit and run the input file for the simulation in ABAQUS. The user should have an understanding of the input file format and how to edit it to suit their needs. It is recommended to use caution and consult with experts in the field when using the code for research or clinical purposes.

# Getting Started
These instructions will get you a copy of the project up and running on your local machine for development and testing purposes. 
To get started with this project, you will need to have the following software installed on your local machine:
•	Python 3.x
•	Gmsh (version 4.x)
•	ABAQUS (version 6.x or higher) / FE *
•	Pentacam elevation files in csv format for the patients you wish to generate models for
•	Surgical data file 
You will also need to have the patient_data.csv file in the same directory as the script which contains patient id & surgical parameters and the respective pentacam elevation file in csv format named as patient id
Once you have these dependencies installed, you can run the script by calling the generate_model() function and passing in a list of patient ids for which you want to generate models.
# File Structure
It is important to note that the patient_information list must contain the following information in the specified order: patient_id, surgical_zone, ablation_zone, sphere, cylinder, angle, curvature, distance between attachments and displacement to be applied on model.
# Prerequisites
The following packages are required to run the script:
•	Numpy 
•	math
•	gmsh
•	sys
•	matplotlib
•	trimesh
•	os
You can install these packages by running the following command in your terminal:
Code example: 
`pip install numpy math gmsh sys matplotlib trimesh os`

# How does it work?
The generate_model() function is the main function of the program that generates a finite element model of a patient-specific lenticule mesh. It takes in a patient_information list as its input and uses it to generate the model. The function is composed of several smaller functions that each performs a specific task in the model generation process.
Here is an overview of how the generate_model() function works:
1.	The function starts by initializing the gmsh library and adding a new model to the library.
code
    gmsh.initialize() gmsh.model.add("lens") 
2.	The function then reads the patient's pentacam elevation data using the read_pentacam() function and returns a point cloud of the anterior and posterior surface of the cornea.
code
points_front, points_back = read_pentacam(patient_id) 
3.	Next, the function removes the epithelium thickness from the elevation data using the remove_epithelium() function.
code
points_middle = remove_epithelium(points_front, points_back) 
4.	The function then calculates and models the ablation profile using the mrochen() function.
code
mrochen(points_middle, surgical_zone, ablation_zone, sphere, cylinder, angle, curvature) 
5.	The function then uses the holes_in_lenticule() function to add holes to the lenticule, where the hooks of the uniaxial test attach to the sample. This function must be adapted to your loading situation
code
holes_in_lenticule(holes_dist) 
6.	The function then uses the identify_surfaces() function to identify the surfaces of the complete model and the mesh_model() function to mesh the model.
code
surface_ids = identify_surfaces() mesh_model() 
7.	Once the meshing is done, the function writes the Abaqus input file for the model to a folder called PatientInput using the write_Abaqus() function.
code
filename = os.pardir + "/PatientInput/" + patient_id + '.inp' write_Abaqus(filename, surface_ids, disp) 
8.	Finally, the function finalizes the gmsh library and ends.
code
gmsh.finalize() 

# Code Examples

Use of the ‘generate_model’ function 
Code
# Import necessary packages
import numpy as np
import math
import gmsh
import sys
import matplotlib.tri as mtri
import trimesh
import os

# Example usage of the generate_model function
patient_information = ['Patient1', 6.0, 8.0, -5.0, 0.0, 90.0, -3.0, 4,0.4 ]
generate_model(patient_information)

Here each field in patient information corresponds to: 
1.	patient_id 
2.	surgical_zone 
3.	ablation_zone 
4.	sphere 
5.	cylinder 
6.	angle 
7.	radius of curvature 
8.	Distance between loading hooks
9.	Loading on the sample – in this case displacement 

Use of Abaqus Input File function
The code writes the Abaqus input file using the write_abaqus_input function. This function takes in the necessary parameters for the simulation, such as the model geometry, boundary conditions, and material properties, and formats them into the correct format for the Abaqus input file.
To customize the input file for your specific needs, you can edit the following parts of the code:
1.	Model geometry: The model geometry is defined in the model_geometry variable. You can edit this variable to change the dimensions or shape of the model.
2.	Boundary conditions: The boundary conditions are defined in the boundary_conditions variable. You can edit this variable to change the type or magnitude of the boundary conditions.
3.	Material properties: The material properties are defined in the material_properties variable. You can edit this variable to change the properties of the material.
4.	Abaqus options: The options for the Abaqus simulation are defined in the abaqus_options variable. You can edit this variable to change the options for the simulation, such as the type of analysis or the time step.
It is important to note that the format of the input file is specific to Abaqus, and changing any parameter may also require changes in other sections of the code.
# Troubleshooting
Typical problems may occur when
1. Pentacam file formats may change
2. Version changes on Python or GMSH

Please get in touch with the authors for any necessary clarifications
# Citation
Any reproduction or parts of this code must be cited as __.
