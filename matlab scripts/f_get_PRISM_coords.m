function [qx_mesh, qy_mesh, q_dist] = f_get_PRISM_coords(cell_dim,real_pixel_size,int_f)
%returns fourier space coords of prismatic simulation with interpolation
%factor int_f, with applied fftshift 
    
    %if inputs are singular, assume symmetry
    if length(cell_dim) < 2
        cell_dim(2) = cell_dim(1);
    end
    
    if length(real_pixel_size) < 2
        real_pixel_size(2) = real_pixel_size(1);
    end
    
    if length(int_f) < 2
        int_f(2) = int_f(1);
    end
    
    %calculate image and final pixel sizes
    fx = int_f(1);
    fy = int_f(2);
    f_x = 4*fx;
    f_y = 4*fy;
    im_size(1) = f_x*round(cell_dim(1)/(f_x*real_pixel_size(1)));
    im_size(2) = f_y*round(cell_dim(2)/(f_y*real_pixel_size(2)));

    pixel_size = cell_dim./im_size;
    
    %get indices for q-coords
    qxInd = getInd(im_size(1));
    qyInd = getInd(im_size(2));
    
    %get coordinates for q-space
    qx = getCoords(im_size(1),pixel_size(1));
    qy = getCoords(im_size(2),pixel_size(2));
    
    %make q-grid, interpolate with indices, and reduce by f respective to
    %dimensions of f
    %result is fftshifted
    [qx_mesh,qy_mesh,q_dist] = makeGrid(qx,qxInd,qy,qyInd,[fx fy]);
    

end

function q_ind = getInd(im_size)
    q_ind = zeros(im_size/2,1);
    
    n_1 = im_size;
    n_quarter1 = im_size/4;
    for i = 1:n_quarter1
        q_ind(i) = i;
        q_ind(i+n_quarter1) = (i-n_quarter1)+n_1;
    end
end

function q = getCoords(im_size,pixel_size)
    
    q = zeros(im_size,1);
    nc = floor(im_size/2);
    dp = 1/(im_size*pixel_size);
    for i = 0:1:im_size-1
        q(mod(nc+i,im_size)+1) = (i-nc)*dp; 
    end

end

function [qxa_out_red,qya_out_red,qdist] = makeGrid(qx,qxInd,qy,qyInd,f)
    [qxa,qya] = meshgrid(qx,qy);

    qxa_out = zeros(length(qxInd),length(qxInd));
    qya_out = qxa_out;
    
    %apply indexing to qmeshes
    for y = 1:length(qyInd)
        for x = 1:length(qxInd)
           qxa_out(y,x) = qxa(qyInd(y),qxInd(x));
           qya_out(y,x) = qya(qyInd(y),qxInd(x)); 
        end
    end

    %reduce them
    qxa_out_red = qxa_out(1:f(2):end,1:f(1):end);
    qya_out_red = qya_out(1:f(2):end,1:f(1):end);
    
    %calculate distance
    qdist = sqrt(qxa_out_red.^2+qya_out_red.^2);
    
    %apply fftshifts
    qxa_out_red = fftshift(qxa_out_red);
    qya_out_red = fftshift(qya_out_red);
    qdist = fftshift(qdist);
end
