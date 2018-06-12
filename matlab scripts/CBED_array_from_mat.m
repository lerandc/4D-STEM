clearvars
files = dir('*.mat');

base_name = 'pic';
base_ext = '_FP1.mat';
sub_im_test = loadImageFromMat(strcat(base_name,'_X0_Y0',base_ext));
sub_im_size = size(sub_im_test);
final_size = 60*sub_im_size;
final_img = zeros(final_size);

for i = 0:59
    for j = 0:59
        f_name = strcat(base_name,'_X',num2str(j),'_Y',num2str(i),base_ext);
        current_img = loadImageFromMat(f_name);
        s_x = i*sub_im_size(2)+1; f_x = (i+1)*sub_im_size(2); 
        s_y = j*sub_im_size(1)+1; f_y = (j+1)*sub_im_size(1);
        final_img(s_y:f_y,s_x:f_x) = rot90((current_img),2);
    end
end


function img = loadImageFromMat(f_name)
    data = load(f_name);
    fields = fieldnames(data);
    img = data.(fields{1});
end
