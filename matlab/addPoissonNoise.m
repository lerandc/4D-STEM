%%Script for applying Poisson noise to simulated PACBED images
%%A functional version of this script was made later to facilate one shot processing of the raw CBEDs
%%Author: Luis Rangel DaCosta, lerandc@umich.edu
%%Last comment date: 8-2-2018

scale_factor = [1 10 50 100 150 250].*1e4; %scale to add poisson noise
%base loop for iterating through files
files = dir('*.mat');
tic
for i = 1:100
    for j = 1:length(scale_factor)
        orig = loadImageFromMat(files(i).name);
        noisy = addNoise(orig,scale_factor(j));
        %saveNoiseArray(files(i).name,noisy,scale_factor(i));
    end
end
toc

function img = loadImageFromMat(f_name)
    data = load(f_name);
    fields = fieldnames(data);
    img = data.(fields{1});
end

function saveNoiseArray(base_name,noisePACBED,scale)
    end_name = strcat(base_name(1:end-4),'_noise',num2str(scale/1e4));
    save(end_name,'noisePACBED');
end

function noisy = addNoise(orig,scale)
    noisy = poissrnd(scale.*orig)./scale;
end