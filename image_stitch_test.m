clearvars
close all
files = dir('*_FP1.mrc');

for i = 1:length(files)
    img_name = files(i).name;
    tomo = mrcReader(img_name);
    mat = tomo.stack;
    mat = fftshift(mat);
    mat = sqrt(mat);
    min_val = min(min(mat)); 
    max_val = max(max((mat-min_val)));
    new_img = 256*(mat-min_val)./max_val; %normalize and scale to 0-256
    new_img = uint8(new_img); %new_img = new_img(108:138,108:138);
    imwrite(new_img,parula(256),strcat(img_name(1:end-4),'.tif'));
end

files = dir('*_FP1.tif');
len = length(files);
name_cell = cell(len,1);
count = 1;
for Y = 0:1:59
    for X = 0:1:59
        name_cell{count} = strcat('6_4_f_4_multislice_comp_X',num2str(X),'_Y',num2str(Y),'_FP1.tif');
        count = count+1;
    end
end

montage(name_cell,'ThumbnailSize',[75 75])