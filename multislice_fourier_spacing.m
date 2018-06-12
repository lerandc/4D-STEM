%fourier spacing for PRISM multislice 
clearvars
m = 9.19383e-31;
e = 1.602177e-19;
c = 299792458;
h = 6.62607e-34;
E0 = 200000;
lambda = 1e10*(h/(sqrt(2*m*e*E0)))/(sqrt(1+(e*E0)/(2*m*c*c)));

real_pixel = 0.1;
cell_dim = 48.8125;
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

dpx = 1/(im_size*real_pixel);
ncx = floor(im_size/2);
qmax = dpx*ncx/2;
alpha_max = 1e3*qmax*lambda;

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
