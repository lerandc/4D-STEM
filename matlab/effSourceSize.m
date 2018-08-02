%%This script applies the blurring effect caused by finite electron source size by applying a convolution of the CBED arrays in real space
%%A functional form of this script was later made to enable one shot processing of CBED images
%%Author: Luis Rangel DaCosta, lerandc@umich.edu
%%Last comment date: 8-2-2018

clearvars
close all

base_name = '6_4_multislice2';
base_ext = '_FP1.mat';
img_map = getPrismMap(59,59);
sim_pixel_size = 10; %pm
source_size = 90; %90 pm FWHM


imsize = size(loadImageFromMat(strcat(base_name,'_X0_Y0',base_ext)));
imsize2 = size(img_map);

center = floor(imsize2/2)+1;
[xmesh,ymesh] = meshgrid(1:imsize2(2),1:imsize2(1));
ymesh = (ymesh-center(1));
xmesh = (xmesh-center(2));

sigma = (source_size/sim_pixel_size)./(2.355);

kernel = getGaussKernel2D(sigma,xmesh,ymesh);
kernel2 = fspecial('gaussian',[60 60], sigma);

tic
orig_CBED_array = get4DArray(base_name,base_ext,img_map,imsize);
toc
%%
tic
result = convolve2D(orig_CBED_array,kernel2);
aligned_result = alignResult(result,img_map);
toc

%%
tic
save4DArray(aligned_result,base_name,base_ext);
toc

%%
%manual comparison of CBEDs pre and post convolution
%{
f = imagesc(zeros(imsize));
title('original');
h = gcf;
figure;
f2 = imagesc(zeros(imsize));
h2 = gcf;
title('convolved');

for i = 1:60
    for j = 1:60
        f.CData = permute(orig_CBED_array(i,j,:,:),[3 4 1 2]);
        figure(h);
        caxis([0 max(max(permute(orig_CBED_array(i,j,:,:),[3 4 1 2])))]);
        f2.CData = permute((aligned_result(i,j,:,:)),[3 4 1 2]);
        figure(h2);
        caxis([0 max(max(permute((aligned_result(i,j,:,:)),[3 4 1 2])))]);
        pause(0.1)
        drawnow;
    end
end

%}
%%

function img = loadImageFromMat(f_name)
    data = load(f_name);
    fields = fieldnames(data);
    img = rot90(data.(fields{1}),2);
end

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

function array = get4DArray(base_name,base_ext,map,imsize)
    %create empty array of X x Y x kx x ky 
    array = zeros(cat(2,size(map),imsize));
    for i = 1:size(map,1)
        for j = 1:size(map,2)
            ID = map{i,j};
            fname = strcat(base_name,'_X',num2str(ID(1)),...
                '_Y',num2str(ID(2)),base_ext);
            array(i,j,:,:) = (loadImageFromMat(fname));
        end
    end
end

function save4DArray(result,base_name,base_ext)
    
    for i = 0:size(result,1)-1
        for j = 0:size(result,2)-1
            f_name = strcat(base_name,'_blur90_X',num2str(j),...
                '_Y',num2str(i),base_ext);
            cbed = result(i+1,j+1,:,:);
            save(f_name,'cbed')
        end
    end
end

function kernel = getGaussKernel2D(sigma,xmesh,ymesh)
    gx = @(x,y) (1/(2*pi*sigma*sigma)).*exp(-(x.^2+y.^2)/(2*sigma*sigma));
    kernel = gx(xmesh,ymesh);
end

function result = convolve2D(CBED_array,kernel)
    f_kernel = fft2(kernel);
    result = CBED_array;
    
    for k = 1:size(CBED_array,3)
        for l = 1:size(CBED_array,4)                           
            result(:,:,k,l) = (ifft2(f_kernel.*((fft2(CBED_array(:,:,k,l))))));
        end
    end

end

function aligned = alignResult(CBED_array,map)
    new_map = fftshift(map);
    aligned = zeros(size(CBED_array));
    for i = 1:60
        for j = 1:60
            ID = new_map{i,j};
            aligned(i,j,:,:) = CBED_array(ID(2)+1,ID(1)+1,:,:);
        end
    end

end

