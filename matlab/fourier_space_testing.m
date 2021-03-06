clearvars
m = 9.19383e-31;
e = 1.602177e-19;
c = 299792458;
h = 6.62607e-34;
E0 = 300e3;
lambda = 1e10*(h/(sqrt(2*m*e*E0)))/(sqrt(1+(e*E0)/(2*m*c*c)));

real_pixel_size = 0.01325;
cell_dim = 58.4749;
z_size = 347.65;
slice_thickness = 1.9525;
f = 4;
f_x = 4*f;
im_size = f_x*round(cell_dim/(f_x*real_pixel_size));

pixel_size = cell_dim/im_size;

dpx = 1/(im_size*real_pixel_size);
ncx = floor(im_size)/2;
qmax = dpx*ncx/2;
alpha_max = 1e3*qmax*lambda;
d_step = 1;
d_vec = d_step/2 : d_step : alpha_max-d_step/2;
Ndet = ceil(alpha_max);
N_streams = 4;
N_gpu = 4;
total_streams = N_gpu*N_streams;
batch_size = 4;
im_size = im_size*(1/1);

N_planes = floor(z_size/slice_thickness) + (mod(z_size,slice_thickness) ~= 0);

est_memory_device_single = (im_size*im_size)*(8+8*N_planes+8+4+4+4+...
    N_streams*(batch_size*8+batch_size*4))+(N_streams*(Ndet*4));

est_memory_device_stream = (im_size*im_size)*(8+8+4+4+4+...
    N_streams*(8+batch_size*8+batch_size*4))+N_streams*((Ndet*4));

est_memory_host = (im_size*im_size)*(8+8*N_planes+8+4+4+4)+...
    total_streams*(Ndet*4);

actual_host = 40004247168;
actual_stream = 5350889376;
actual_single_xfer = 44081092512;

if im_size > 2000
    manual_override = 1;
    if ~manual_override
        error('reset override before you crash your laptop thanks')
    end
end
qxInd = zeros(im_size/2,1);

n_1 = im_size;
n_quarter1 = im_size/4;

for i = 1:n_quarter1
    qxInd(i) = i;
    qxInd(i+n_quarter1) = (i-n_quarter1)+n_1;
end
qyInd = qxInd;


res = zeros(im_size,1);
nc = floor(im_size/2);
dp = 1/(im_size*pixel_size);
for i = 0:1:im_size-1
    res(mod(nc+i,im_size)+1) = (i-nc)*dp; 
end

qx = res;
qy = res';

[qxa,qya] = meshgrid(qx,qy);

qxa_out = zeros(length(qxInd),length(qxInd));
qya_out = qxa_out;

for y = 1:length(qyInd)
    for x = 1:length(qxInd)
       qxa_out(y,x) = qxa(qyInd(y),qxInd(x));
       qya_out(y,x) = qya(qyInd(y),qxInd(x)); 
    end
end

qxa_out_s = fftshift(qxa_out);
qya_out_s = fftshift(qya_out);

%reduce them
qxa_out_red = qxa_out(1:f:end,1:f:end);
qya_out_red = qya_out(1:f:end,1:f:end);

q_final_coords = cell(size(qxa_out_red));
q_final_coords_dist = zeros(size(qxa_out_red));

for i = 1:size(qxa_out_red,1)
    for j = 1:size(qxa_out_red,2)
        q_final_coords{i,j} = [qxa_out_red(i,j) qya_out_red(i,j)];
        q_final_coords_dist(i,j) = sqrt(qxa_out_red(i,j).^2 +...
                                    qya_out_red(i,j).^2);
    end
end

q_final_coords_s = fftshift(q_final_coords);
q_final_coords_dist_s = fftshift(q_final_coords_dist);