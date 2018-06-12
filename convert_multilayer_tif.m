clearvars
close all

files = dir('*.tif');
for i = 1:length(files)
    img_name = files(i).name;
    mat = imread(img_name,2);
    mat = sqrt(mat);
    new_img = 256*(mat-(min(min(mat))))./max(max(mat)); %normalize and scale to 0-256
    new_img = uint8(new_img);
    imwrite(new_img,parula(256),strcat(img_name(1:end-4),'.tif'));
end