function [qx_mesh,qy_mesh,q_dist,q_mask] = f_get_multislice_coords(cell_dim,real_pixel)
%Returns the fourier space (un shifted) coordinates and anti-aliasing mask
%for multislice algorithm as implemented in Prismatic.
%NEED TO UPDATE TO ACCOUNT FOR NON SQUARE SIMULATION CELLS
    f_x = 16;
    im_size = f_x*round(cell_dim/(f_x*real_pixel));
    pixel_size = cell_dim/im_size;

    q = zeros(im_size,1);
    nc = floor(im_size/2);
    dp = 1/(im_size*pixel_size);
    for i = 0:1:im_size-1
        q(mod(nc+i,im_size)+1) = (i-nc)*dp; 
    end

    qx = q; qy = q;

    [qxa,qya] = meshgrid(qx,qy);

    % dpx = 1/(im_size*real_pixel);
    % ncx = floor(im_size/2);
    % qmax = dpx*ncx/2;
    % alpha_max = 1e3*qmax*lambda;

    q2 = qxa.^2+qya.^2;
    q1 = sqrt(q2);

    qmask = zeros(im_size,im_size);
    offset_x = size(qmask,2)/4;
    offset_y = size(qmask,1)/4;
    ndimx = size(qmask,2);
    ndimy = size(qmask,1);

    for y = 0:1: ndimy/2-1
        for x = 0:1: ndimx/2-1
            mod1 = mod(mod(y-offset_y,ndimy)+ndimy,ndimy);
            mod2 = mod(mod(x-offset_x,ndimx)+ndimx,ndimx);
            qmask(mod1+1,mod2+1) = 1;
        end
    end
    
    qx_mesh = qxa; qy_mesh = qya;
    q_dist = q1;
    q_mask = qmask;
end
