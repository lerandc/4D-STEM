%crop IGOR outputs, delete orig images, save as MRC 
clearvars
close all

cell_dim_x = 48.8125;
cell_dim_y = 48.8125;

files = dir('*.tif');

[qx_mesh,qy_mesh,q_dist] = f_get_IGOR_coords(cell_dim_x,cell_dim_y,files(1).name);

%mask for cutting of distance in Fourier space to around zero order disk
cut_off_dist = 1.125;
mask = q_dist < cut_off_dist; 


for iter = 1:length(files)
    f_name = files(iter).name;
    img = imread(f_name,2); %intensity data in 2nd layer of IGOR tiffs
    masked_img = img.*mask; %zeros data cut_off_dist away in Fourier space
    cropped_img = imageCrop(masked_img); %removes zero padding from matrix
    %mat files are lowest storage of dlm,mat,and mrc
    save(strcat(f_name(1:end-4),'.mat'),'cropped_img','-v7');
end

fclose('all');

for i=1:length(files)
    delete(files(i).name)
end
