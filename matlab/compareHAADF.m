%%Error calculation script for comparing autostem HAADF images and prismatic HAADF images
%%Author: Luis Rangel DaCosta, lerandc@umich.edu
%%Last comment date: 8-2-2018

clearvars
close all

mse_labels = {'none','corr','rigid','sub'}';

%set reference image and test image folders
ref_folder = 'C:\Users\leran\Desktop\Simulations and Data\6-13\IGOR HAADF data\';
test_folder = 'C:\Users\leran\Desktop\Simulations and Data\6-14\3D scale tests\';
ref_HAADF_name = 'STO_HAADFstack.mat';
test_HAADF_name = 'HAADF_f8.mat';

ref_HAADF_data = load(strcat(ref_folder,ref_HAADF_name));
test_HAADF_data = load(strcat(test_folder,test_HAADF_name));

%IGOR HAADF is double float, sliced into layers
ref_final_layer = ref_HAADF_data.HAADF(:,:,end);
ref_final_layer = single(ref_final_layer);

%prismatic is already single float, should only have one layer but
%including end reference for future conversion if necessary
test_final_layer = test_HAADF_data.HAADF(:,:,end);

%debugging
% imagesc(log(ref_final_layer)); colormap(gray); colorbar
% figure;
% imagesc(log(test_final_layer)); colormap(gray); colorbar

%no registration
mse1 = immse(test_final_layer,ref_final_layer);

%cross correlation
mse2 = crossCorrelate(test_final_layer,ref_final_layer);

%rigid registration
[optimizer,metric] = imregconfig('monomodal');
[test_img] = imregister(test_final_layer,ref_final_layer,'rigid',optimizer,metric);
mse3 = immse(test_img,ref_final_layer);

%sub pixel rigid registration
error = 1; %dummy values to start loop
diff_error = 2;
pixel_frac = 1;
%loop for sub-pixel error count to converge once fine enough sub-pixel has
%been reached
while(diff_error>1e-3)
    pixel_frac = pixel_frac+1;
    [dft_out, test_img2] = dftregistration(fft(ref_final_layer),fft(test_final_layer),pixel_frac);
    mse4 = immse(ifft(test_img2),ref_final_layer);
    diff_error = abs(error-mse4)./error;
    error = mse4;
end

mse_table = table(mse1, mse2, mse3, mse4,'VariableNames',mse_labels);
figure; imagesc(ref_final_layer); colormap(gray);
caxis([0 max(max(ref_final_layer))]); colorbar;
title('Reference HAADF'); fig1 = gcf;

figure; imagesc(test_final_layer); colormap(fig1.Colormap);
caxis([0 max(max(ref_final_layer))]); colorbar;
title('Test HAADF'); fig2 = gcf;
save('HAADF_compare_results.mat','mse_table','fig1','fig2','ref_HAADF_data','test_HAADF_data');

function mse = mask_immse(im1,im2,mask)
    N = sum(sum(mask));
    mse = sum(sum((im1-im2).^2))/N;
end

function mse = crossCorrelate(test_im,ref_im)
    corr = xcorr2(ref_im,test_im);
    [~,max_ind] = max(abs(corr(:)));
    [y_max,x_max] = ind2sub(size(corr),max_ind);
    im_size = size(ref_im);
    offset = [y_max-im_size(1),x_max-im_size(2)];

    if any(offset)
        empty_mat = zeros(im_size(1)+abs(offset(1)),im_size(2)+abs(offset(2)));
        test_mat = empty_mat; ref_mat = empty_mat;

        %accounting for negative offsets creates weird logic
        neg = offset < 0;
        pos = ~neg;

        ref_mat(1-neg(1)*offset(1):im_size(1)-neg(1)*offset(1),1-neg(2)*offset(2):im_size(2)-neg(2)*offset(2)) = ref_im;
        test_mat(1+pos(1)*offset(1):end+neg(1)*offset(1),1+pos(2)*offset(2):end+neg(2)*offset(2)) = test_im;

        %FOR NOW MANUALLY CHECK ALIGNMENT
        mask = ((test_mat>0)+(ref_mat>0))==2;
        mse = mask_immse(ref_mat,test_mat,mask);
    else
        mse = immse(ref_im,test_im); 
    end
end