%%Converts raw MRC from prismatic multislice algorithm into matlab binary files
%%Newer scripts to process the outputs on the cluster made this obsolete
%%Author: Luis Rangel DaCosta, lerandc@umich.edu
%%Last comment date: 8-2-2018
clearvars
close all

cell_dim = 48.8125;
real_pixel = 0.023834;

[qxa,qya,qdist,qmask] = f_get_multislice_coords(cell_dim,real_pixel);

files = dir('*FP*.mrc');
FP_check = 0; %bool check to do FP averaging
n_FP = 4;
cut_off_dist = 1.125; %cut off dist in fourier space
dist_mask = qdist < cut_off_dist;

if ~FP_check
    for i = 1:length(files)
       fname = files(i).name;
       map = mrcReader(fname);
       map = map.stack;
       out_map = imageCrop(fftshift(dist_mask.*map));
       save(strcat(fname(1:end-4),'.mat'),'out_map','-v7');
    end
else
    file_cell = struct2cell(files)'; file_cell = file_cell(:,1);
    sort_files = regexp(file_cell,'.*_FP','match','once');
    sort_files = unique(sort_files);
    sort_files = sort_files(~cellfun('isempty',sort_files));
    check_for_4D = contains(sort_files,'_X');
    base_names = sort_files(check_for_4D);
    wait =-1;
end

fclose('all');
% 
% for i = 1:length(files)
%     delete(files(i).name) 
% end
