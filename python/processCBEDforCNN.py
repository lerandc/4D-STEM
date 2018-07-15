import util4D
import numpy as np
import scipy.io as sio
import os
import sys
from joblib import Parallel, delayed

def prepareCBED(N,sub_folder):
    cell_dim = (48.9125,48.9125) #cell dimensions of simulation cell (take from prismatic output)
    real_pixel = (0.06,0.06) #size of pixel used to sample potential space
    probe_step = (20,20) #PM
    E0 =  200e3 #accelerating energy of electron beam
    q_cut = 40 #mrad, radius from center of zero disk
    q_cut_style = 'rect' #style of cut off window
    algorithm = 'm' # 'm' for multislice and 'p' for prism

    folder = '/srv/home/lerandc/outputs/712_STO/'+sub_folder+'/' #folder convention initial folder with sub folders given in command line call
    #read in FP averaged CBED outputs and crop them in k-space 
    util4D.processCBED(cell_dim=cell_dim,real_pixel=real_pixel,E0=E0,q_cut=q_cut,q_cut_style=q_cut_style,algorithm=algorithm,folder=folder,slice=N)

    print('processed CBEDs for slice: '+str(N))
    source_size = 90 #PM, size of finite electron source
    base_name = 'STO_thick_FP16_tilt_'+sub_folder+'_slice' #beginning part of regular file name 
    base_ext = '_FPavg_crop.npy' #end part of regular file name
    radii = (5,6,7) #integration radii
    centers = util4D.genCenters([(9,9)],1) #centers and offset shift [-i,i] used to serve as center in real space to integrate CBED into PACBED
    out_name = 'Sr_PACBED'
    out_ext = '.npy'
    array_size = (16,16) #size of real space array of CBED images

    cur_base_name = base_name+str(N)
    cur_out_ext = '_'+str(N)+out_ext
    
    #apply blur caused by finite size of electron source by convoluting CBED array in real space with guassian filter
    cbedArray = util4D.effSourceSize(source_size,probe_step,cur_base_name,base_ext,array_size,folder)
    print('applied effective source size for slice: ' + str(N))
    #create 
    util4D.createPACBED(radii,centers,out_name,cur_out_ext,array_size,cbedArray,folder)
    print('created PACBEDs for slice: '+str(N))

    #scaling factors used to set lambda level of poisson distribution when applying noise
    scale_factors = [10, 50, 100]
    pacbeds = [pacbed for pacbed in os.listdir(folder) if (('Sr_PACBED' in pacbed) and ('_'+str(N)+'.npy') in pacbed)]
    for pacbed in pacbeds:
        tmp = np.load(folder+pacbed)
        for scale in scale_factors:
                noisy = util4D.addPoissonNoise(tmp,scale)
                np.save(folder+pacbed[:-4]+'_noise'+str(scale),noisy)
    
if __name__ == '__main__':
    inList =[]
    sub_folder = sys.argv[1]
    #range input is number of slices in output
    #sub folder is generated in command line call to designate target folder
    for i in range(52):
        inList.append([i,sub_folder])

    #process CBED -> PACBED flow in parallel over n_job cores
    #MSCData nodes have 48 cores and ~190 GB ram
    #node overhead is minimal, so unless processing huge, huge arrays, should not worry about overhead in ram of utilziing this many cores
    n_jobs = 40
    if len(inList) < 40:
        n_jobs = len(inList)
    Parallel(n_jobs=n_jobs,timeout=999999)(delayed(prepareCBED)(N=N,sub_folder=s) for N,s in inList) 