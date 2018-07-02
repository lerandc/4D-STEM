clearvars
m = 9.19383e-31;
e = 1.602177e-19;
c = 299792458;
h = 6.62607e-34;
E0 = 300e3;
lambda = 1e10*(h/(sqrt(2*m*e*E0)))/(sqrt(1+(e*E0)/(2*m*c*c)));

real_pixel_size = 0.03;
cell_dim = 65;
f = 2;
f_x = 4*f;
im_size = f_x*round(cell_dim/(f_x*real_pixel_size));

pixel_size = cell_dim/im_size;

dpx = 1/(im_size*real_pixel_size);
ncx = floor(im_size)/2;
qmax = dpx*ncx/2;
alpha_max = 1e3*qmax*lambda;

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
