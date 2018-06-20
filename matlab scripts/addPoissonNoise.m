scale_factor = [1 10 50 100 150 250].*1e4; %scale to add poisson noise
%base loop for iterating through files
files = dir('*.mat');
for i = 1:5 %length(files)
    for j = 1:length(scale_factor)
        orig = loadImageFromMat(files(i).name);
        noisy = addNoise(orig,scale_factor(i));
        saveNoiseArray(files(i).name,noisy,scale_factor(i));
    end
end

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