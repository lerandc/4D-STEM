#Host script for generating convolution images for FCC Cu 111
#Author: Luis Rangel DaCosta, lerandc@umich.edu
#Last comment date: 7-31-2018

from read_xyz import read_xyz
from STEM_mod import STEM
import matplotlib.pyplot as plt
import ase
import numpy as np
from ase.build import fcc111

def removeAtoms(atoms,z_lim):
    #returns an ASE Atoms object with atoms above a certain z position removed
    #atoms is the original ASE atoms object
    #z_lim is a float for the desired max depth, in Angstroms
    newAtoms = [atom for atom in atoms if atom.position[2] < z_lim]
    return ase.Atoms(newAtoms)

def readHAADFfromMRC(fname):
    #reads mrc files according to mrc2014 file standard
    #based loosely on matlab version of openNCEM mrcReader(https://github.com/ercius/openNCEM, https://bitbucket.org/ercius/openncem)
    #ideally,  could be easily expanded to work for arbitrary MRC files, but works minimally for prismatic output
    f = open(fname, 'rb')
    count = 10
    b = f.read(count*4)
    data_types = {0:np.uint8,1:np.intc,2:np.float32,6:np.uint16}
    c = np.frombuffer(b, dtype=np.int32,count=count)
    data_size = c[:3]
    data_type = data_types[c[3]]

    f.seek(1024)

    count = data_size[0]*data_size[1]*data_size[2]
    cbed = f.read(count*4)
    cbed_arr = np.frombuffer(cbed,dtype=data_type,count=count)
    cbed_arr = np.reshape(cbed_arr,tuple(data_size))
    f.close
    f.closed
    return cbed_arr

new_model = False

if new_model:
    #generate new Cu 111 FCC model from scratch (probably more accurate)
    atoms = fcc111('Cu',size=(24,28,170),a=3.615,orthogonal=True)
else:
    #get atoms object using ASE and pre defined Cu 111 FCC model
    atoms = read_xyz('FCC_Cu_111_350nm.xyz',format='xyz')

probe_step=.088052 #angstroms
resolution = 1/probe_step
HWHM = 0.4
z_list = []
for i in range(21):
    z_list.append(256.5+i*4.5)

switch = False
#iterate through the same thicknesses as previously simulated FCC Cu 111 simulations in multisilice
for z in z_list:
    #remove atoms from atom s object
    newAtoms = removeAtoms(atoms,z)

    #create a generator object so that convolution images can be generated using STEM_mod from structopt
    positions = newAtoms.get_positions()
    dimensions = [np.max(positions[:,0]),np.max(positions[:,1])]
    parameters = {'HWHM':HWHM,'dimensions':dimensions,'resolution':resolution}
    generator = STEM(parameters=parameters)
    image = generator.get_image(newAtoms)

    #add dimension to image so that they can be stacked
    image = image[:,:,np.newaxis]

    #create an image stack if the first time through loop with first input
    #otherwise, add on to previously existing stack
    if switch:
        image_array = np.dstack((image_array,image[280:350,280:350,:]))
    else:
        image_array = image[280:350,280:350,:]


    switch = True
    
np.save('Cu_111_convolution',image_array)


