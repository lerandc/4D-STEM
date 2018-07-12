import os
import scipy.io as sio
import scipy.misc as smisc
import numpy as np
import re as regexp
import joblib
import math

def loadImageFromMat(fname):                
    #assumes that mat file only contains an image matrix
    img_array = sio.loadmat(fname)
    fields = sio.whosmat(fname)
    img_array = img_array[fields[0][0]]
    return img_array

def genCenters(centers,shiftRange):
    #center is list of tuples
    shifts = []
    for s in shiftRange:
        shifts.append((s,s))

    outCenters = []
    for center in centers:
        for shift in shifts:
            outCenters.append(tuple(map(lambda x,y:x+y, center, shift)))

    return list(set(outCenters))

def addPoissonNoise(orig,scale):
    #orig is numpy array
    orig = orig * scale
    noisy = np.random.poisson(orig)
    return noisy

def prismMap(xlim, ylim):
    #xlim and ylim the sizes of the CBED array 
    pMap = [[[y,x] for x in range(xlim)] for y in range(ylim)]
    return pMap

def get4Darray(base_name,base_ext,map_lim,imsize):
    #loads images into npy array
    #imsize should be a tuple
    #map_lim should be a tuple of (xlim,ylim)
    #base name and base_ext are strings corresponding to the regular file names prismatic outputs
    #ex: base_name = '73_test_', base_ext = '_FPavg.npy'

    out_sz = map_lim + imsize
    output = np.zeros(out_sz,dtype=np.float32)
    for x in range(map_lim[0]):
        for y in range(map_lim[1]):
            output[x,y,:,:] = np.load(base_name+'_X'+str(x)+'_Y'+str(y)+base_ext)

    return output

def createPACBED(radii,centers,base_name,base_ext,out_name,out_ext,array_size):
    imsize = (np.load(base_name+'_X0_Y0'+base_ext)).shape

    origCBEDarray = get4Darray(base_name,base_ext,array_size,imsize)
    origCBEDarray = np.reshape(origCBEDarray,np.prod(array_size)+imsize)
    xlim,ylim = array_size
    xm,ym = np.meshgrid(range(1,xlim+1),range(1,ylim+1))

    for radius in radii:
        for cent_x,cent_y in centers:
            xm_cent = xm-cent_x
            ym_cent = ym-cent_y
            dist = np.sqrt(xm_cent**2 + ym_cent**2)

            pacbed = np.zeros(imsize,dtype=np.float32)
            mask = reshape(dist < radius,-1)
            for i in range(len(mask)):
                if mask[i]:
                    pacbed += orig_CBED_array[i,:,:]
            pacbed /= sum(mask)

            f_name = out_name+'_'+str(cent_x) + '_' + str(cent_y) + '_' + str(radius) + out_ext
            np.save(f_name,pacbed)

def effSourceSize(source_size,pixel_size,base_name,base_ext,array_size):
    imsize = (np.load(base_name+'_X0_Y0'+base_ext)).shape
    origCBEDarray = get4Darray(base_name,base_ext,array_size,imsize)
    sigma = (source_size/pixel_size)/(2.355)
    kernel = 1

    return convolve2D(origCBEDarray,kernel)

def convolve2D(CBEDarray,kernel):
    fkernel = np.fft.fftshift(kernel)
    kx,ky = CBEDarray.shape[-2:]
    result = CBEDarray
    for k in range(kx):
        for l in range(ky):
            result[:,:,k,l] = np.fft.ifft2(fkernel*np.fft.fft2(CBEDarray[:,:,k,l]))

    #finish by aligning the result
    return np.fft.fftshift(result,axes=(0,1))
    
def getCoordsM(cell_dim,real_pixel):
    #cell_dim is either single x=y or list of cell dimensions x,y
    #real pixel is sampling of potential space as single x=y or list of x,y
    if len(cell_dim) < 2:
        cell_dim.append(cell_dim[0])

    if len(real_pixel) < 2:
        real_pixel.append(real_pixel[0])

    im_size = 16*round(cell_dim[0] / (16*real_pixel[0]))
    im_size.append(16*round(cell_dim[1] / (16*real_pixel[1])))

    pixel_size = cell_dim[0] / im_size[0]
    pixel_size.append(cell_dim[1] / im_size[1])

    qx = qCoords(im_size[0],pix_size[0])
    qy = qCoords(im_size[1],pix_size[1])

    qxa,qya = np.meshgrid(qx,qy)
    qdist = map(lambda x,y: (x**2 + y**2)**(0.5), qxa,qya)

    qmask = np.zeros((im_size[1],im_size[0]))
    offset_x = im_size[0] / 4
    offset_y = im_size[1] / 4
    ndimx = im_size[0]
    ndimy = im_size[1]

    for y in range(ndimy/2):
        for x in range(ndimx/2):
            mod1 = (((y - offset_y) % ndimy + ndimy) % ndimy)
            mod2 = (((x - offset_x) % ndimx + ndimx) % ndimx)
            q_mask[mod1,mod2] = 1

    output = {"qxa":qxa,"qya":qya,"qdist":qdist,"qmask":qmask}
    return output

def getCoordsP(cell_dim,real_pixel,int_f):
    if len(cell_dim) < 2:
        cell_dim.append(cell_dim[0])

    if len(real_pixel) < 2:
        real_pixel.append(real_pixel[0])

    if len(int_f) < 2:
        int_f.append(int_f[0])

    fx = int_f[0]
    fy = int_f[1]
    f_x = 4 * fx
    f_y = 4 * fy
    im_size = f_x * round(cell_dim[0]/(real_pixel[0] * f_x))
    im_size.append(f_y * round(cell_dim[0]/(real_pixel[0] * f_y)))

    pixel_size = cell_dim[0] / im_size[0]
    pixel_size.append(cell_dim[1] / im_size[1])

    qxInd = qInd(im_size[0])
    qyInd = qInd(im_size[1])

    qx = qCoords(im_size[0])
    qy = qCoords(im_size[1])

    return makeGrid(qx,qxInd,qy,qyInd,[fx,fy])

def qInd(im_size):
    q_ind = np.zeros((im_size/2,1))

    n = im_size
    n_quarter = im_size / 4
    for i in range(n_quarter):
        q_ind[i] = i + 1;
        q_ind[i+n_quarter-1] = (i-n_quarter)+n+1

    return q_ind

def processCBED():
    #takes raw CBED output and crops it to deisred distance in k-space
    a = 1

def getLambda(E0):
    m = 9.19383e-31
    e = 1.602177e-19
    c = 299792458
    h = 6.62607e-34

    return 1e10*(h/(math.sqrt(2*m*e*E0)))/(math.sqrt(1+(e*E0)/(2*m*c*c)))

def averageScheme():
    a =1

def qCoords(im_size,pix_size):
    q = np.zeros((im_size,1))
    nc = math.floor(im_size/2)
    dp = 1/(im_size * pix_size)
    for i in range(im_size):
        q[int((nc+i) % im_size + 1)] = (i - nc) * dp

    return q

def makeGrid(qx,qxInd,qy,qyInd,f):
    qxa,qya = np.meshgrid(qx,qy)
    qxa_out = np.zeros((len(qxInd),len(qxInd)))
    qya_out = qxa_out

    for y in range(len(qyInd)):
        for x in range(len(qxInd)):
            qxa_out[y,x] = qxa[qyInd[y],qxInd[x]]
            qya_out[y,x] = qya[qyInd[y],qxInd[x]]

    qxa_out_red = qxa_out[0:-1:f[1],0:-1:f[0]]
    qya_out_red = qya_out[0:-1:f[1],0:-1:f[0]]
    qdist = map(lambda x,y:(x**2+y**2)**(0.5),qxa_out_red,qya_out_red)

    output = {"qxa_out_redexit":qxa_out_red,"qya_out_red":qya_out_red,"qdist":qdist}
    return output

def gaussKernel(sigma,imsize):
    x,y = np.meshgrid(range(1,imsize+1),range(imsize+1))
    x = x - imsize//2
    y = y - imsize//2
    tmp = -(x**2+y**2)/(2*sigma**2)
    return (1/(2*np.pi*sigma**2))*np.exp(tmp)