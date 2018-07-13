import util4D
import numpy as np
import scipy.io as sio
import os
from joblib import Parallel, delayed
from math import sqrt

def prepareCBED(N):
    cell_dim = (48.9125,48.9125)
    real_pixel = (0.06,0.06)
    E0 =  200e3
    q_cut = 40 #mrad, radius from center of zero disk
    q_cut_style = 'rect'
    algorithm = 'm'
    folder = '/srv/home/lerandc/outputs/712_STO/1_0/'
    util4D.processCBED(cell_dim=cell_dim,real_pixel=real_pixel,E0=E0,q_cut=q_cut,q_cut_style=q_cut_style,algorithm=algorithm,folder=folder,slice=N)
    print('processed CBEDS for slice: '+str(N))
    source_size = 90 #pm, FWHM
    base_name = 'STO_thick_FP16_tilt_1_0_slice'
    base_ext = '_FPavg_crop.npy'
    radii = (5,6,7)
    centers = util4D.genCenters([(9,9)],1)
    out_name = 'Sr_PACBED'
    out_ext = '.npy'
    array_size = (16,16)

    cur_base_name = base_name+str(N)
    cur_out_ext = '_'+str(N)+out_ext
    
    cbedArray = util4D.effSourceSize(source_size,real_pixel,cur_base_name,base_ext,array_size,folder)
    print('applied effective source size for slice: ' + str(N))
    util4D.createPACBED(radii,centers,out_name,cur_out_ext,array_size,cbedArray,folder)
    print('created PACBEDs for slcie: '+str(N))

    scale_factors = [10, 50, 100]
    pacbeds = [pacbed for pacbed in os.listdir(folder) if (('Sr_PACBED' in pacbed) and ('_'+str(N)+'.npy') in pacbed)]
    for pacbed in pacbeds:
        tmp = np.load(folder+pacbed)
        for scale in scale_factors:
                noisy = util4D.addPoissonNoise(tmp,scale)
                np.save(folder+pacbed[:-4]+'_noise'+str(scale),noisy)
    
if __name__ == '__main__':
    nSlices =[]
    for i in range(52):
        nSlices.append(i)

    print(nSlices)
    n_jobs = 40
    if len(nSlices) < 40:
        n_jobs = len(nSlices)
    Parallel(n_jobs=n_jobs,timeout=999999)(delayed(prepareCBED)(N) for N in nSlices) 