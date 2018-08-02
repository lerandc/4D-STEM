%%Plotting script to look at individual CBED patterns one by one
%%Author: Luis Rangel DaCosta, lerandc@umich.edu
%%Last comment date: 8-2-2018

%CBED script
clearvars
close all
base_name = '6_1_GPUtest8';
ext = '.mrc';

for FP = 1
    FPstr = strcat('_FP',num2str(FP));
    figure(1)
    clf
    for i = 0:16
        for j = 0:16
            position = strcat('_X',num2str(i),'_Y',num2str(j));
            fname = strcat(base_name,position,FPstr,ext);
            axes('position',[0.0588*i 0.0588*j 0.0588 0.0588])
            %axes('OuterPosition',[0.0588*i 0.0588*j 0.0625 0.0625])
            tomo = mrcReader(fname);
            mat = tomo.stack;
            mat = fftshift(mat);
            imagesc(sqrt(mat))
            colormap(gray(256))
            axis equal off
        end
    end
end

pbaspect([1 1 1])
fig = gcf;
fig.Position = ([0 0 1000 1000]);
