clearvars

cell_dim_x = 48.8125;
cell_dim_y = 48.8125;
qx_pixel = 1/(cell_dim_x);
qy_pixel = 1/(cell_dim_y);

file_name = 'STO_im000_cbed-0-0-7.tif';
img = imread(file_name,2);

im_size = size(img);

qx_vec = (1:im_size(2)).*qx_pixel;
qy_vec = (1:im_size(1)).*qy_pixel;

size_chk_x = mod(im_size(2),2);
size_chk_y = mod(im_size(1),2);

zero_pos = [0 0];

if size_chk_x
    zero_pos(2) = ceil(im_size(2)/2);
else
    zero_pos(2) = im_size(2)/2+1;
end

if size_chk_y
    zero_pos(1) = ceil(im_size(1)/2);
else
    zero_pos(1) = im_size(1)/2+1;
end

qx_vec = qx_vec - qx_vec(zero_pos(2));
qy_vec = qy_vec - qy_vec(zero_pos(1));

[qx_mesh,qy_mesh] = meshgrid(qx_vec,qy_vec');

q_dist = sqrt(qx_mesh.^2+qy_mesh.^2);
