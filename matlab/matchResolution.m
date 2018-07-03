function [re_img1,re_img2] = matchResolution(img1,img2)
%Resizes larger image to resolution of smaller image. Images must be
%similar rectangles.
    img1_size = size(img1); ratio1 = img1_size(1)/img1_size(2);
    img2_size = size(img2); ratio2 = img2_size(1)/img2_size(2);
    
    if ratio1 ~= ratio2
        error(strcat('Images are not of the same geometry, ',...
             'check orientation and size of input images.'))
    end
    
    %resize larger image to smaller resolution and scale values to maintain
    %constant sum of values
    if img1_size(1) > img2_size(1)
        re_img1 = imresize(img1,img2_size);
        re_img1 = re_img1.*(img1_size(1)/img2_size(1))^2;
        re_img2 = img2;
    else
        re_img1 = img1;
        re_img2 = imresize(img2,img1_size);
        re_img2 = re_img2.*(img2_size(1)/img1_size(1))^2;
    end
end
