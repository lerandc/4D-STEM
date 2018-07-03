clearvars
close all

FP = 4; %num of phonon configurations
X = 59; %largest x grid pos
Y = 59; %largest y grid pos
base = '6_5_p_FP_4';
ext = '.mrc';
first_file = strcat(base,'_X0_Y0_FP1.mrc');
map = mrcReader(first_file);
get_size = size(map.stack);

%iterate by CBED position, then sum over FP
for i = 0:1:X
    for j = 0:1:Y
        position = strcat('_X',num2str(i),'_Y',num2str(j));
        average = zeros(get_size);
        for f = 1:1:FP
            FPstr = strcat('_FP',num2str(f));
            file_name = strcat(base,position,FPstr,ext);
            map = (mrcReader(file_name));
%             imagesc(fftshift(map.stack)); xlim([90 160]); ylim([90 160])
%             drawnow
            average = average+map.stack;
        end
        average = fftshift(average./FP);
        figure; imagesc(average); xlim([90 160]); ylim([90 160])
        min_val = min(min(average));
        max_val = max(max((average-min_val)));
        new_img = 256*(average-min_val)./max_val; %normalize and scale to 0-256
        new_img = uint8(new_img); %new_img = new_img(108:138,108:138);
        imwrite(new_img,parula(256),strcat(file_name(1:end-7),'avg.tif'));
    end
end
