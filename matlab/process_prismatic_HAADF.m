%%Script to create arbitrary vritual detectors from 3D prismatic output
%%Author: Luis Rangel DaCosta, lerandc@umich.edu
%%Last comment date: 8-2-2018
clearvars
close all

%folder = 'C:\Users\leran\Desktop\Simulations and Data\6-13\613_3D_scale\';
file_name = 'slice7_STO_layer_compare_m.mrc';

min_ang = 88;
max_ang = 440; 
d_step = 1; %mrad

if ~exist('folder') %#ok<*EXIST>
    map = mrcReader(file_name);
else
    map = mrcReader(strcat(folder,file_name));
end

stack = map.stack;
stack = permute(stack,[2 3 1]);
[HAADF,out_angmax] = integrateBins(min_ang,max_ang,d_step,stack);
out_angmin = min_ang;

if ~exist('folder')
    save('HAADF.mat','HAADF','out_angmin','out_angmax');
else
    save(strcat(folder,'HAADF.mat'),'HAADF','out_angmin','out_angmax');
end

function [integrated, angmax] = integrateBins(angmin,angmax,d_step,stack)
    dims = size(stack);
    
    alpha_max = dims(3)*d_step;
    init = round(angmin/d_step);
    final = round(angmax/d_step);
    
    if alpha_max < angmax
        integrated = sum(stack(:,:,init:end),3);
        angmax = alpha_max;
    else
        integrated = sum(stack(:,:,init:final),3);
    end
end