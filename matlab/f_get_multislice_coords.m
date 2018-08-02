%%Function to get fourier space coordinates of 4D images as implemented in prismatic multislice
%%Author: Luis Rangel DaCosta, lerandc@umich.edu
%%Last comment date: 8-2-2018 as

function [qx_mesh,qy_mesh,q_dist,q_mask] = f_get_multislice_coords(cell_dim,real_pixel)
%Returns the fourier space (un shifted) coordinates and anti-aliasing mask
%for multislice algorithm as implemented in Prismatic.
    f_x = 16;
    f_y = 16;

    if length(cell_dim) < 2
        cell_dim(2) = cell_dim(1);
    end
    
    if length(real_pixel) < 2
        real_pixel(2) = real_pixel(1);
    end

    im_size(1) = f_x*round(cell_dim(1)/(f_x*real_pixel(1)));
    im_size(2) = f_y*round(cell_dim(2)/(f_y*real_pixel(2)));

    pixel_size = cell_dim./im_size;
    
    qx = getCoords(im_size(1),pixel_size(1));
    qy = getCoords(im_size(2),pixel_size(2));

    [qx_mesh,qy_mesh] = meshgrid(qx,qy);
    qx_mesh = fftshift(qx_mesh);
    qy_mesh = fftshift(qy_mesh);
    q_dist = fftshift(sqrt(qx_mesh.^2+qy_mesh.^2));
    
    q_mask = zeros(im_size(2),im_size(1));

    offset_x = size(q_mask,2)/4;
    offset_y = size(q_mask,1)/4;
    ndimx = size(q_mask,2);
    ndimy = size(q_mask,1);

    for y = 0:1: ndimy/2-1
        for x = 0:1: ndimx/2-1
            mod1 = mod(mod(y-offset_y,ndimy)+ndimy,ndimy);
            mod2 = mod(mod(x-offset_x,ndimx)+ndimx,ndimx);
            q_mask(mod1+1,mod2+1) = 1;
        end
    end
    
    q_mask = fftshift(q_mask);
end

function q = getCoords(im_size,pixel_size)
    
    q = zeros(im_size,1);
    nc = floor(im_size/2);
    dp = 1/(im_size*pixel_size);
    for i = 0:1:im_size-1
        q(mod(nc+i,im_size)+1) = (i-nc)*dp; 
    end

end
