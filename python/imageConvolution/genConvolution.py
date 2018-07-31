from read_xyz import read_xyz
from STEM_mod import STEM
import matplotlib.pyplot as plt
import ase
import numpy as np
from ase.build import fcc111

def removeAtoms(atoms,z_lim):
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

pos = atoms.get_positions()
#print(np.max(pos[0,:]),np.max(pos[:,1]),np.max(pos[:,2]))
#print(np.min(pos[0,:]),np.min(pos[:,1]),np.min(pos[:,2]))

probe_step=.088052 #angstroms
resolution = 1/probe_step
HWHM = 0.4
z_list = []
for i in range(21):
    z_list.append(256.5+i*4.5)

switch = False
for z in z_list:
    newAtoms = removeAtoms(atoms,z)
    positions = newAtoms.get_positions()
    dimensions = [np.max(positions[:,0]),np.max(positions[:,1])]#,np.max(positions[:,2])]

    parameters = {'HWHM':HWHM,'dimensions':dimensions,'resolution':resolution}
    generator = STEM(parameters=parameters)
    image = generator.get_image(newAtoms)
    image = image[:,:,np.newaxis]
    #print(image.shape)
    if switch:
        image_array = np.dstack((image_array,image[280:350,280:350,:]))
        #print(image_array.shape)
    else:
        image_array = image[280:350,280:350,:]
        #print(image_array.shape)

    #print(z)
    #print(image_array.shape)
    switch = True
    
np.save('Cu_111_convolution',image_array)


