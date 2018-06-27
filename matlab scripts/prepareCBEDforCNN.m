clearvars
close all
%%
cell_dim = [48.8125, 48.8125];
real_pixel = [0.025];
E0 = 200e3;
FP = 1;
behavior = 'unique';
q_cut = 50; %mrad
q_cut_style = 'rect';
algorithm = 'm';

f_processCBED(cell_dim,real_pixel,E0,FP,behavior,q_cut,q_cut_style,algorithm);

%%
source_size = 90; %pm, FWHM
base_name = 'test';
base_ext = '.mat';
array_size = [60,60]; %+1 of max indices of 4D array

f_effSourceSize(source_size,real_pixel,base_name,base_ext,array_size);

%%
radii = [5 6 7]; %pixels
centers = generateCenters([35 35],[1 1]); %centers of atom
out_name = 'Sr_PACBED';
out_ext = '.mat';

f_createPACBED(radii,centers,base_name,base_ext,out_name,out_ext,array_size);
%%
file_list = dir(strcat(out_name,'*',out_ext));
scale_factors = [1 5 10 15 25];

f_addPoissonNoise(file_list,scale_factors);


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



        


