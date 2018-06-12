% clearvars
load('indexing_arrays.mat');

% base_name = '6_4_multislice2';
% base_ext = '_FP1.mat';
 
base_name = 'STO_im';
base_ext = '-7.mat';

COM_x = zeros(60,60); COM_y = zeros(60,60);
X = 1:109;
Y = 1:109;


for i = 1:60
    for j = 1:60
    %fID = prism_map{i,j};
    fID = igor_map{i,j};
    %f_name = strcat(base_name,'_X',num2str(fID(1)),'_Y',num2str(fID(2)),base_ext);
    f_name = strcat(base_name,threeDigit(fID(3)),'_cbed-',num2str(fID(1)),'-',num2str(fID(2)),base_ext);
    patch = loadImageFromMat(f_name);
    %patch = rot90(patch,-1);
    
    xsum = sum(patch,1); % use original uncropped image to calculate COM
    ysum = sum(patch,2);

    xsum(xsum<0) = 0;   % set negative entries to zero
    ysum(ysum<0) = 0;   % negative entries come from imresize, are very close to zero

    xwmean = wmean(X,xsum);
    ywmean = wmean(Y,ysum');

    COM_x(i,j) = xwmean;
    COM_y(i,j) = ywmean;

    end
end

function img = loadImageFromMat(f_name)
    data = load(f_name);
    fields = fieldnames(data);
    img = data.(fields{1});
end

function str = threeDigit(int)
%creates 3 digit length string for int from 0 to 999
    if int < 0 || int > 999
        error('Number cannot be expressed in three integer digits.')
    elseif int < 10
        str = strcat('00',num2str(int));
    elseif int < 100
        str = strcat('0',num2str(int));
    else
        str = num2str(int);
    end
end