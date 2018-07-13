import util4D
import numpy as np
import scipy.io as sio
import os

cell_dim = (48.9125,48.9125)
real_pixel = (0.06,0.06)
E0 =  200e3
q_cut = 40 #mrad, radius from center of zero disk
q_cut_style = 'rect'
algorithm = 'm'
folder = 'C:/Users/Luis/Desktop/Research/UW REU 2018/processing_test_folder/'
"""
qdict = util4D.getCoordsM(cell_dim,real_pixel)
print(qdict.keys())
qxa = qdict['qxa']
qya = qdict['qya']
qdist = qdict['qdist']
qmask = qdict['qmask']
print(qxa.shape,qya.shape,qdist.shape,qmask.shape)


#util4D.processCBED(cell_dim=cell_dim,real_pixel=real_pixel,E0=E0,q_cut=q_cut,q_cut_style=q_cut_style,algorithm=algorithm,folder=folder)

source_size = 90 #pm, FWHM
base_name = 'STO_thick_FP16_tilt_0_0_slice'
base_ext = '_FPavg_crop.npy'
radii = (5,6,7)
centers = util4D.genCenters([(7,7)],1)
out_name = 'Sr_PACBED'
out_ext = '.npy'
array_size = (16,16)
n_slice = 1

for slice in range(n_slice):
    cur_base_name = base_name+str(slice)
    cur_out_ext = '_'+str(slice)+out_ext
    
    cbedArray = util4D.effSourceSize(source_size,real_pixel,cur_base_name,base_ext,array_size)
    util4D.createPACBED(radii,centers,out_name,cur_out_ext,array_size,cbedArray)

"""
scale_factors = [1, 5, 10, 50, 100]

pacbeds = [pacbed for pacbed in os.listdir(folder) if 'Sr_PACBED' in pacbed]

for pacbed in pacbeds:
    tmp = np.load(pacbed)
    for scale in scale_factors:
            noisy = util4D.addPoissonNoise(tmp,scale)
            np.save(pacbed[:-4]+'_noise'+str(scale),noisy)
