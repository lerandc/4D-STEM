%%Converts raw MRC from prismatic PRISM algorithm into matlab binary files
%%Newer scripts to process the outputs on the cluster made this obsolete
%%Author: Luis Rangel DaCosta, lerandc@umich.edu
%%Last comment date: 8-2-2018

clearvars
close all

tic

cell_dim = 48.8125;
real_pixel = 0.025;
f = 4;
[qxa,qya,qdist] = f_get_PRISM_coords(cell_dim,real_pixel,f);

files = dir('*FP*.mrc');
FP_check=0; %bool check to do FP averaging
n_FP=16;
cut_off_dist = 2; %cut off dist in fourier space
dist_mask = qdist < cut_off_dist;


if ~FP_check
    for i = 1:length(files)
       fname = files(i).name;
       map = mrcReader(fname);
       map = fftshift(map.stack);
       out_map = imageCrop((dist_mask.*map)).*(f^4);
       %save(strcat(fname(1:end-4),'.mat'),'out_map','-v7');
    end
else
    file_cell = struct2cell(files)'; file_cell = file_cell(:,1);
    sort_files = regexp(file_cell,'.*_FP','match','once');
    sort_files = unique(sort_files);
    sort_files = sort_files(~cellfun('isempty',sort_files));
    check_for_4D = contains(sort_files,'_X');
    base_names = sort_files(check_for_4D);
    
    test_img = mrcReader(strcat(base_names{1},'1.mrc'));
    test_img = fftshift(test_img.stack);
    im_size = size(test_img);
    toc
    
    for i = 1:length(base_names)
        result = getAvgFP(base_names{i},n_FP,im_size,f);
        out_map = imageCrop((dist_mask.*result));
        save(strcat(base_names{i},'_avg.mat'),'out_map','-v7');
    end
end
    
fclose('all');

toc

% for i = 1:length(files)
%     delete(files(i).name) 
% end

function avg = getAvgFP(f_name,FP,im_size,f)
    avg = zeros(im_size);
    for i = 1:FP
       map = mrcReader(strcat(f_name,num2str(FP),'.mrc'));
       avg = avg+map.stack;
    end
    avg = fftshift(avg./FP).*(f^4);
end
