clearvars
close all

base_name = '6_4_multislice2';
base_ext = '_FP1.mat';
%get mapping array
prismMap = getPrismMap(59,59);

%generate masking coordinates
xvec = 1:size(prismMap,2);
yvec = 1:size(prismMap,1);
[xmesh, ymesh] = meshgrid(xvec,yvec);

%get center, shift coordinates,and define a distance function
center = [30.5, 30.5]; %manually defined using HAADF, Y,X
xmesh = xmesh-center(2);
ymesh = ymesh-center(1);
dist = sqrt(xmesh.^2 + ymesh.^2);

%create mask
radius = 7; %pixels
mask = dist <= radius;

%shift mask and map of images to column to make loops easier
mask = reshape(mask,[],1);
prismMap = reshape(prismMap,[],1);

pacbedMap = prismMap(mask);

pacbed = zeros(109);

for iter = 1:length(pacbedMap)
    ID = pacbedMap{iter};
    fname = strcat(base_name,'_X',num2str(ID(1)),'_Y',num2str(ID(2)),base_ext);
    pacbed = pacbed + rot90(loadImageFromMat(fname),2);
end

pacbed = pacbed./(length(pacbedMap));

function map = getPrismMap(X_lim,Y_lim)
%X lim is highest X value of prism imaging
%Y lim is highest Y value of prism imaging
    map = cell(Y_lim+1,X_lim+1);
    for Y = 0:1:Y_lim
        for X = 0:1:X_lim
            map{Y+1,X+1} = [X Y];
        end
    end
end

function img = loadImageFromMat(f_name)
    data = load(f_name);
    fields = fieldnames(data);
    img = data.(fields{1});
end