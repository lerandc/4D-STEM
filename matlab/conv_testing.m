%%This script tried to improve quality of interpolated images/replicating interpolated images by applying some convolutions
%%Author: Luis Rangel DaCosta, lerandc@umich.edu
%%Last comment date: 8-2-2018

clearvars
close all
map = mrcReader('C:\Users\leran\Desktop\Simulations and Data\6-13\613_tile_large_pot_bound\p_bound_5.mrc');
haadf = map.stack;
haadf = permute(haadf,[2 3 1]);
haadf = sum(haadf(:,:,88:end),3);
load('STO_HAADFstack.mat');
HAADF = single(HAADF);

final_haadf = convolveHAADF(haadf,1,1);

figure; imagesc(HAADF(:,:,8));
ulim = max(max(HAADF(:,:,8)));
colorbar; caxis([0 ulim])
title('IGOR')

figure; imagesc(haadf);
colorbar; caxis([0 ulim])
title('orig f4')

figure; imagesc(final_haadf);
colorbar; caxis([0 ulim])
title('f4 after gaussian convolution and weights')

test = convolveHAADF2(haadf,1,1);
test2 = convolveHAADF3(haadf,1,1);

figure; imagesc(test);
colorbar; caxis([0 ulim])
title('mulitple peaks')

test3 = weightMatrix(haadf);

figure; imagesc(test3)
colorbar; caxis([0 ulim])
title('weight matrix')

mse1 = immse(HAADF(:,:,8),haadf)
mse2 = immse(HAADF(:,:,8),final_haadf)
mse3 = immse(HAADF(:,:,8),test)
mse4 = immse(HAADF(:,:,8),test3)
close all
h = deconvwnr(HAADF(:,:,8),double(haadf),1);
b = conv2(haadf,(h),'same');
mse5 = immse(HAADF(:,:,8),abs(b));

function corrected = weightMatrix(orig)
    weights = [1/16 1/4 1/16;
        1/4 1 1/4;
        1/16 1/4 1/16];
    
    field = zeros(size(orig)+2);
    field(2:end-1,2:end-1) = orig;
    weight_matrix = zeros(size(orig));
    
    for i = 2:size(orig,1)+1
        for j = 2:size(orig,2)+1
            weight_matrix(i-1,j-1) = sum(sum(field(i-1:i+1,j-1:j+1).*weights));
        end
    end
    
    weight_matrix = weight_matrix./(max(max(weight_matrix)));
    
    orig_sum = sum(sum(orig));
    corrected = weight_matrix.*orig;
    corrected_sum = sum(sum(corrected));
    scale = orig_sum/corrected_sum;
    
    corrected = corrected.*scale;

end

function corrected = convolveHAADF(orig,std_dev,order)

    gauss_curve_right = normpdf(linspace(0,10,30),0,std_dev);
    gauss_curve = [fliplr(gauss_curve_right) gauss_curve_right];

    orig_sum = sum(sum(orig));
    
    conv_res_x = conv2(orig,gauss_curve,'same');
    conv_res_y = conv2(orig,gauss_curve','same');

    weights_x = (conv_res_x-min(min(conv_res_x)))/(max(max(conv_res_x-min(min(conv_res_x)))));
    weights_y = (conv_res_y-min(min(conv_res_y)))/(max(max(conv_res_y-min(min(conv_res_y)))));
    

    new_haadf = (orig.*(weights_x.^(order))).*(weights_y.^(order));
    new_haadf = new_haadf+min(min(orig));
    
    new_sum = sum(sum(new_haadf));
    scale = orig_sum/new_sum;

    corrected = new_haadf.*scale;
end

function corrected = convolveHAADF2(orig,std_dev,order)

    gauss_curve_right = normpdf(linspace(0,5,30),0,std_dev);
    gauss_curve = [fliplr(gauss_curve_right) gauss_curve_right];
    
    for i = 1:1
        peaks = islocalmax(orig(:,i));
        gauss_conv = addOffsetVectors(gauss_curve,find(peaks));

    end
        conv_res_y = conv2(orig,gauss_conv,'same');
                
    for i = 1:1
        peaks = islocalmax(orig(i,:))';
        gauss_conv = addOffsetVectors(gauss_curve',find(peaks));

    end
    
        conv_res_x = conv2(orig,gauss_conv,'same');

    orig_sum = sum(sum(orig));

    weights_x = (conv_res_x-min(min(conv_res_x)))/(max(max(conv_res_x-min(min(conv_res_x)))));
    weights_y = (conv_res_y-min(min(conv_res_y)))/(max(max(conv_res_y-min(min(conv_res_y)))));
    

    new_haadf = (orig.*(weights_x.^(order))).*(weights_y.^(order));
    new_haadf = new_haadf+min(min(orig));
    
    new_sum = sum(sum(new_haadf));
    scale = orig_sum/new_sum;

    corrected = new_haadf.*scale;
end

function corrected = convolveHAADF3(orig,std_dev,order)

    gauss_curve_right = normpdf(linspace(0,10,30),0,std_dev);
    gauss_curve = [fliplr(gauss_curve_right) gauss_curve_right];
    
    conv_res_x = conv2((orig),gauss_curve,'same');
    conv_res_y = conv2((orig),gauss_curve','same');
    
    orig_sum = sum(sum(orig));

    weights_x = conv_res_x;
    weights_y = conv_res_y;
    

    new_haadf = (orig.*(weights_x.^(order))).*(weights_y.^(order));
    new_haadf = new_haadf+min(min(orig));
    
    new_sum = sum(sum(new_haadf));
    scale = orig_sum/new_sum;

    corrected = new_haadf.*scale;
end

function new_vec = addOffsetVectors(orig,offsets)
%adds a vector in the offset locations
    new_vec = zeros(size(orig));
    half = ceil(size(orig,2)/2);
    
    for i = 1:length(offsets)
        if offsets(i) < half
            new_vec = new_vec+circshift(orig,-abs(offsets(i)-half));
        else
            new_vec = new_vec+circshift(orig,abs(offsets(i)-half));
        end
    end
end