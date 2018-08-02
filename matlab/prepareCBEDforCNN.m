%%Host script to take raw prismatic output and then run it through a processing pipeline to be ready for CNN training
%%Author: Luis Rangel DaCosta, lerandc@umich.edu
%%Last comment date: 8-2-2018 as

clearvars
close all
tic
%%
cell_dim = [48.9125, 48.9125];
real_pixel = [0.075];
E0 = 200e3;
FP = 1;
behavior = 'unique';
q_cut = 50; %mrad
q_cut_style = 'rect';
algorithm = 'm';

%f_processCBED(cell_dim,real_pixel,E0,FP,behavior,q_cut,q_cut_style,algorithm,1);

%%
source_size = 90; %pm, FWHM
base_name = 'STO_thick_025_025_slice';
base_ext = '_FP1.mat';

radii = [5 6 7]; %pixels
centers = generateCenters([7 7],[1 1]); %centers of atom
out_name = 'Sr_PACBED';
out_ext = '.mat';
array_size = [13,13]; %+1 of max indices of 4D array
n_slices = 51;

for slice = 0:n_slices
    cur_base_name = strcat(base_name,num2str(slice));
    cur_out_ext = strcat('_',num2str(slice),out_ext);
    
    f_effSourceSize(source_size,real_pixel,cur_base_name,base_ext,array_size);
    f_createPACBED(radii,centers,cur_base_name,base_ext,out_name,cur_out_ext,array_size);
end
%%
toc
tic
file_list = dir(strcat(out_name,'*',out_ext));
scale_factors = [1 5 10 50 100]; %loosely proportional to beam current/camera frequency

f_addPoissonNoise(file_list,scale_factors);

toc
%%

function centers = generateCenters(center_list,shifts)
    %shifts is 2 element vecor with max y_shift in pixels, max x_shift in
    %pixels
        yvec = -shifts(1):1:shifts(1);
        xvec = -shifts(2):1:shifts(2);
        [mesh, ~] = meshgrid(xvec,yvec);
    
        centers = zeros(numel(mesh)*size(center_list,1),2);
        
        count = 1;
        for i = 1:size(center_list,1)
            for j = yvec
                for k = xvec
                    centers(count,:) = [center_list(i,1)+j, center_list(i,2)+k];
                    count = count+1;
                end
            end
        end
    
end



        


