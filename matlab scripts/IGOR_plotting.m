clearvars
close all
base_name = 'STO_im'; cbed = '_cbed-';
name_cell = cell(3600,2);
layer = 7;
ending = strcat('-',num2str(layer),'.tif');
count = 1;

for i = 0:1:224
    for j = 0:1:3
        for k = 0:1:3
            if i < 10
                i_name = strcat('00',num2str(i));
            elseif i >= 10 && i < 100
                i_name = strcat('0',num2str(i)); 
            else
                i_name = num2str(i);
            end
            
            name_cell{count,1} = strcat(base_name,i_name,cbed,...
                num2str(j),'-',num2str(k),ending);
            name_cell{count,2} = j+240*mod(i,15)+4*floor(i/15)+60*k;
            count = count+1;
        end
    end
end

[~,order] = sort(cell2mat(name_cell(:,2)));

ordered_names = cell(3600,1);
for iter = 1:3600
    ordered_names{iter} = name_cell{order(iter)};
end

montage(ordered_names,'ThumbnailSize',[75 75])
