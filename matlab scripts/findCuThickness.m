%%
%set up reference and test arrays
clearvars
close all

files = dir('*.mrc'); files = {files.name}';
base = 'multislice_2Doutput_slice';
ext = '_FCC_Cu_111_350nm_8FP.mrc';

sort_cell = regexp(files,strcat(base,'\d*+'),'match','once');
file_cell = IDfiles(sort_cell);
test_im_size = getImSize(strcat(file_cell{1},ext));

sigma = (90/8.8052)/2.355;
gauss_kernel = fspecial('gaussian',[test_im_size(1) test_im_size(2)], sigma);
%for 3D output
% annular_range = [73,350]; d_step = 1; %mrads for both
% test_array = processTestImages(file_cell,ext,gauss_kernel,im_size,annular_range,d_step);

test_array = processTestImages(file_cell,ext,gauss_kernel,test_im_size);
%test_array2 = processTestImages(file_cell,ext,gauss_kernel,im_size,annular_range,d_step);

ref_array = loadImageFromMat('ref_cell.mat');
ref_im_size = size(ref_array);
%%
%set up masks
%grid for mask for test images
radius = [13 14 15 16 17 18]'; %in pixels for the reference image,manually choose
test_px_size = 0.088052; %pixel step sizes in angstroms
ref_px_size = 0.044026;
test_centers = [18, 20];
test_centers = generateCenters(test_centers,[1.5 1.5]);
ref_centers = [65, 56;
               37, 36;
               33, 70;
               69, 21];
ref_centers = generateCenters(ref_centers,[3 3]);

[test_mask, ref_mask] = setUpMasks(radius,test_px_size,ref_px_size,test_centers,ref_centers,test_im_size,ref_im_size);
%%
%prepare reference data
rough_scale = 3.2862e4;
offset = 9.8761e3;
%they used following (approx)
% rough_scale = 4.189905e4;
% offset = 9.9243e3;

ref_array = (ref_array-offset)./(.7*rough_scale-offset);
ref_sum = sum(sum(ref_array.*ref_mask))./(sum(sum(ref_mask)));
ref_max = max(max(ref_array.*ref_mask));

%%
%compare the images
[best_inds, errors, out_radius] = compareArrays(test_array,ref_array,test_mask,ref_mask,radius);
%%
%error plotting and interpretation
[sorted_inds,ind_order] = sort(best_inds(:,1));
order_error = errors(ind_order,1);
[bins,ia,~] = unique(sorted_inds);
bins_error = zeros(length(bins),1);
ia(end+1) = length(sorted_inds)+1;
for i = 1:length(ia)-1
    bins_error(i) = sum(order_error(ia(i):ia(i+1)-1))./(ia(i+1)-ia(i)+1);
end
figure;
plot(bins,bins_error);
title('mean error of slice')

figure;
histfit(best_inds(:,1),length(ia)-1)
[mu, sig] = normfit(best_inds(:,1));
title(['frequency of best slice, \mu = ' num2str(mu) ', \sigma = ' num2str(sig)])
lb = mu-1.96*sig/(sqrt(length(best_inds)));
ub = mu+1.96*sig/(sqrt(length(best_inds)));

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

function [best_inds,errors,out_radius] = compareArrays(test_array,ref_array,test_mask,ref_mask,radius)
%mask arrays should be ordered in chunks of Y x X x N, where N is number of
%radii times number of centers, and N is radii chunks of centers,
%therefore, should loop through layers first, then centers, then radii; 
%will also have to compare the different groups of centers
%output is: best_ind of [integrated area, max value], correspond error
%list, and the corresponding integration radius
    
    %initialize iteration limits
    rad_iter = size(radius,1);
    ref_center_iter = size(ref_mask,3)./rad_iter;
    test_center_iter = size(test_mask,3)./rad_iter;
    
    %initialize output arrays
    best_inds = zeros(rad_iter*ref_center_iter*test_center_iter,2);
    errors = best_inds;
    out_radius = best_inds(:,1);
    
    count = 1;
    for i = 1:rad_iter
        for j = 1:ref_center_iter %#ok<BDSCI>
            ref_current_mask = ref_mask(:,:,(i-1)*ref_center_iter+j);
            ref_sum = sum(sum(ref_array.*ref_current_mask))./(sum(sum(ref_current_mask)));
            ref_max = max(max(ref_array.*ref_current_mask));
            
            %force test mask loop to use corresponding radius of ref mask
            %loop
            t_start = (i-1)*test_center_iter+1;
            t_end = t_start + test_center_iter-1;
            for k = t_start:t_end
                best_error = [1 1];
                test_current_mask = test_mask(:,:,k);
                
                for layer = 1:size(test_array,3)
                    test_current = test_array(:,:,layer);
                    test_sum = sum(sum(test_current.*test_current_mask))./(sum(sum(test_current_mask)));
                    test_max = max(max(test_current.*test_current_mask));
                    
                    area_error = abs(ref_sum-test_sum)/ref_sum;
                    if area_error < best_error(1)
                        errors(count,1) = area_error;
                        best_inds(count,1) = layer;
                        best_error(1) = area_error;
                    end
                    
                    peak_error = abs(ref_max-test_max)/ref_max;
                    if peak_error < best_error(2)
                        errors(count,2) = peak_error;
                        best_inds(count,2) = layer;
                        best_error(2) = peak_error;
                    end
                    
                end
                out_radius(count) = radius(rad_iter);
                count = count+1;
            end
        end
    end
    

end

function [test_mask,ref_mask] = setUpMasks(radius,test_px_size,ref_px_size,test_centers,ref_centers,test_im_size,ref_im_size)
    %radius is Nx1 list of radii to try
    ref_radius = radius;
    test_radius = floor(radius.*(ref_px_size./test_px_size));
    
    %centers is a Nx2 matrix, where N is number of centers to shift around
    %initialzing mask arrays, 3rd dimension defines unique masks
    test_mask = zeros(test_im_size(1),test_im_size(2),length(radius)*size(test_centers,1));
    ref_mask = zeros(ref_im_size(1),ref_im_size(2),length(radius)*size(ref_centers,1));
    
    %radius list is consistent between both but centers are not
    t_offset = size(test_centers,1);
    r_offset = size(ref_centers,1);
    for radii = 1:length(radius)
        for t_center = 1:size(test_centers,1)
            ind = (radii-1)*t_offset+t_center;
            test_mask(:,:,ind) = maskGrids(test_im_size,test_centers(t_center,:),test_radius(radii));
        end
        
        for r_center = 1:size(ref_centers,1)
            ind = (radii-1)*r_offset+r_center;
            ref_mask(:,:,ind) = maskGrids(ref_im_size,ref_centers(r_center,:),ref_radius(radii));
        end
    end
    

end

function mask = maskGrids(im_size,center,radius)

    xvec = 1:im_size(2); yvec = 1:im_size(1);
    [xgrid,ygrid] = meshgrid(xvec,yvec);
    %shift and create dist map
    xgrid = xgrid - center(2);
    ygrid = ygrid - center(1);
    dist = sqrt(xgrid.^2+ygrid.^2);
    mask = dist < radius;

end

function result = convolve2D(stack,kernel)
%convolves array with specified kernel
    f_kernel = fft2(kernel);
    result = ifft2(((fft2(stack))).*f_kernel);
end

function fileCell = IDfiles(sort_cell)
%assumes file cell is list of sliced images, layer is digits after slice
%get slice string
    IDs = regexp(sort_cell,'(?<=slice)\d+','match','once');
    list = ~cellfun(@isempty,IDs); 
    sort_cell = sort_cell(list); IDs = IDs(list);
    %add one to match matlab indexing and convert back
    IDs = num2cell(cellfun(@str2num,IDs)+1);
    %returns sort cell with IDs in next column
    fileCell = [sort_cell IDs];
end

function testArray = processTestImages(f_name,ext,kernel,im_size,annular_range,d_step)
%loads test images, forces double format, convolves images with gaussian
%kernel
%For processing 3D output, fourth argument is range of annular integration
%and fifth argument is size of bin step
testArray = zeros(im_size(1),im_size(2),length(f_name));

    if nargin < 5
        for i = 1:length(f_name)
            fname = strcat(f_name{i,1},ext);
            map = mrcReader(fname);
            stack = double(map.stack);
            final_stack = fftshift(convolve2D(stack,kernel));
            testArray(:,:,f_name{i,2}) = final_stack;
        end
    else
        angmin = annular_range(1); angmax = annular_range(2);
        
        init = round(angmin/d_step);
        final = min(round(angmax/d_step),im_size(3));
        
        for i = 1:length(f_name)
            fname = strcat(f_name{i,1},ext);
            map = mrcReader(fname);
            stack = permute(double(map.stack),[2 3 1]);
            stack = sum(stack(:,:,init:final),3);
            final_stack = fftshift(convolve2D(stack,kernel));
            testArray(:,:,f_name{i,2}) = final_stack;
        end

    end


end

function im_size = getImSize(f_name)
    map = mrcReader(f_name);
    im_size = size(map.stack);
    if length(im_size) > 2
        im_size = circshift(im_size,2);
    end
end

function img = loadImageFromMat(f_name)
    data = load(f_name);
    fields = fieldnames(data);
    img = data.(fields{1});
end

