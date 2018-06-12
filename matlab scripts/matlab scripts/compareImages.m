clearvars
close all
%Before running this script, check: file directories, annular integration
%ranges, and that all images have been preprocessed to be cut off to the
%same distance in Fourier space.

%set reference image and test image folders
ref_folder = 'C:\Users\leran\Desktop\Simulations and Data\6-5\IGOR Final layer orig\layer7\';
test_folder = 'C:\Users\leran\Desktop\Simulations and Data\6-5\FP_convergence\f16\p_FP_16_f16\';
base_ref_name = 'STO_im';
base_test_name = '6_5_p_FP_16_f16';

%by now all files are .mat
ref_ext = '-7.mat';
test_ext = '_FP_avg.mat';

%get map for X0,Y0 style image labeling (PRISM), origin in top left
X_lim = 59; Y_lim = 59;
prism_map = getPrismMap(X_lim,Y_lim);

%get map for IGOR style image labeling, should be same array as prism
img_num = 224; Y_lim = 3; X_lim = 3;
grid_size = [15 15]; %num of images in each column, row
igor_map = getIgorMap(X_lim,Y_lim,img_num,grid_size);

%reshape to reduce loops
prism_map = prism_map';
prism_map = reshape(prism_map,[],1);
igor_map = reshape(igor_map,[],1);

if length(prism_map) ~= length(igor_map)
    error('The two sets of images are not one-one.')
end

%get better way
im_size = [7,7];
%create annular integration masks and resize it size of largest image
[mask1,mask2,mask3] = annularIntMasks([64, 64]);
mask1 = imresize(mask1,im_size); mask2 = imresize(mask2,im_size);
mask3 = imresize(mask3,im_size);
    
check = 0;
result_matrix = cell(size(prism_map));

for iter = 1:length(prism_map)
    %get images from respective folders
    ref_ID = igor_map{iter};
    %%%THIS HAS TO BE CHANGED EVERY TIME DEPENDING ON FILES
    % or you could use regex
    ref_name = strcat(ref_folder,base_ref_name,threeDigit(ref_ID(3)),'_cbed-',...
        num2str(ref_ID(1)),'-',num2str(ref_ID(2)),ref_ext);
    ref_img = loadImageFromMat(ref_name);
    
    test_ID = prism_map{iter};
    test_name = strcat(test_folder,base_test_name,'_X',num2str(test_ID(1)),...
        '_Y',num2str(test_ID(2)),test_ext);
    test_img = loadImageFromMat(test_name);
    test_img = rot90(test_img,2);
    %resize images so they have same resolution
    [ref_img,test_img] = matchResolution(ref_img,test_img);
    
    %calculate mean squared error
    mse_res = immse(test_img,ref_img);
    
    %calculate structural similarity index  (test, ref)
    ssi_res = ssim(rescale(test_img,0,256),rescale(ref_img,0,256));
    
    %calculate peak signal to noise ratio
    psnr_res = psnr(test_img,ref_img,max(max(ref_img)));
    
    %calculate annular integration intensity (masks are set manually)
    %result is reported as % error from 
    if ~check
        s = input('Did you check the annular integration masks?','s');
        check = 1;
    end
    int_res1 = sum(sum(mask1.*ref_img));
    int_res1 = abs(sum(sum(mask1.*test_img))-int_res1)/int_res1;
    
    int_res2 = sum(sum(mask2.*ref_img));
    int_res2 = abs(sum(sum(mask2.*test_img))-int_res2)/int_res2;
    
    int_res3 = sum(sum(mask3.*ref_img));
    int_res3 = abs(sum(sum(mask3.*test_img))-int_res3)/int_res3;
    
    result_matrix{iter} = [mse_res ssi_res psnr_res int_res1 int_res2 int_res3];
end

means = getMeans(result_matrix);

fprintf('Mean of similarity parameters across CBED array: \n')
fprintf('MSE: %d \n',means(1))
fprintf('SSIM: %d \n',means(2))
fprintf('PSNR: %d \n',means(3))
fprintf('Percent Error, 1st annular integration: %d \n', means(4))
fprintf('Percent Error, 2nd annular integration: %d \n', means(5))
fprintf('Percent Error, 3rd annular integration: %d \n', means(6))

function map = getPrismMap(X_lim,Y_lim)
%X lim is highest X value of prism imaging
%Y lim is highest Y value of prism imaging
    map = cell(Y_lim+1,X_lim+1);
    for Y = 0:1:Y_lim
        for X = 0:1:X_lim
            map{Y+1,X+1} = [X Y];
        end
    end
end

function map = getIgorMap(X_lim,Y_lim,img_num,sub_img)
%X lim is highest X value in img
%Y lim is highest Y value in img
%img_num is highest img number in files
%sub_img is vector of number of images in each column, then row

    len = sqrt((img_num+1)*(X_lim+1)*(Y_lim+1));
    map = cell(len,len);
    for i = 0:1:img_num
        for j = 0:1:X_lim
            for k = 0:1:Y_lim
                X_coord = j+(X_lim+1)*floor(i/sub_img(1))+1;
                Y_coord = k+(Y_lim+1)*mod(i,sub_img(2))+1;
                map{Y_coord,X_coord} = [j k i];
            end
        end
    end
    
end

function img = loadImageFromMat(f_name)
    data = load(f_name);
    fields = fieldnames(data);
    img = data.(fields{1});
end

function str = threeDigit(int)
%creates 3 digit length string for int from 0 to 999
    if int < 0 || int > 999
        error('Number cannot be expressed in three integer digits.')
    elseif int < 10
        str = strcat('00',num2str(int));
    elseif int < 100
        str = strcat('0',num2str(int));
    else
        str = num2str(int);
    end
end

function [mask1, mask2, mask3] = annularIntMasks(im_size)
%creates masks for annular integration

    %set up grid coordinates
    qx = 1:im_size(2); qy = 1:im_size(1);
    if mod(qx,2)
       x_shift = ceil(im_size(2)/2);
    else
       x_shift = im_size(2)/2+1; 
    end
    
    if mod(qy,2)
       y_shift = ceil(im_size(1)/2);
    else
       y_shift = im_size(1)/2+1; 
    end
    
    qx = qx-x_shift; qy = qy-y_shift;
    [qxa,qya] = meshgrid(qx,qy);
    %distance grid, based on center origin of rectangular matrices
    qdist = qxa.^2+qya.^2;
    
    %%%%Input relative distances to serve as integration bounds
    mask1 = qdist < 15;
    mask2 = (qdist >= 15) + (qdist < 60);
    mask3 = ones(im_size);
end

function [means] = getMeans(cell_input)
%gets mean values of all test paramters
    len = length(cell_input);
    means = zeros(size(cell_input{1}));
    for i = 1:len
        means = means+cell_input{i};
    end
    means = means./len;
end
