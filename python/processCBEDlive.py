#Script for processing CBED arrays from raw prismatic MRC output to FP average npy arrays
#Ideally, reduces 4D output of prismatic by factor of number of FP, because prismatic does not
#store 4D output during runtime as to prevent massive file overhead
#This script usually takes a while (~30 minutes for about 300-400K images), and with little effort, could probably be parallelized easily
#Author: Luis Rangel DaCosta, lerandc@umich.edu
#Last comment date: 7-15-2018
#written for Python 3.6.5, using scipy 1.1.0, numpy 1.14.5

import numpy as np
import time
import sys
import os

def main():
    base_name = sys.argv[1]
    folder = sys.argv[2]+'/' #sys.argv[2]
    #read in list of both all CBED patterns and patterns of only the first frozen phonon
    #first frozen phonon pattern used to indentify unique probe positions
    cbed_list_unique = [img for img in os.listdir(folder) if (base_name in img) and ('FP1.mrc' in img)]
    cbed_list_all = [img for img in os.listdir(folder) if ((base_name in img) and ('FP' in img)) and ('.mrc' in img)]

    for pos in cbed_list_unique:
        #grab unique position name for given probe position
        cmp = pos.rsplit('_',1)
        cbed_pos = cmp[0][:]
        tmp_array = np.zeros(readCBEDfromMRC(pos).shape,dtype=np.float32)
        fp = 0
        #integrate all frozen phonons onto temporary array, delete raw files as caclulation proceeds
        for cbed in cbed_list_all:
            if (cbed_pos+'_') in cbed:
                tmp_array += readCBEDfromMRC(cbed)
                os.remove(cbed)
                fp += 1
        tmp_array /= fp
        np.save(cbed_pos+'_FPavg',tmp_array)

def readCBEDfromMRC(fname):
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

if __name__ == '__main__':
    main()