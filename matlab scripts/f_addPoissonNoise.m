function f_addPoissonNoise(file_list,scale_factors)

    parfor f_iter = 1:length(file_list)
        f_name = file_list(f_iter).name;
        for scale = scale_factors
            orig = loadImageFromMat(f_name);
            noisy = addNoise(orig,scale);
            saveNoiseArray(f_name,noisy,scale);
        end
    end

end

function img = loadImageFromMat(f_name)
    data = load(f_name);
    fields = fieldnames(data);
    img = data.(fields{1});
end

function saveNoiseArray(base_name,noisePACBED,scale)
    end_name = strcat(base_name(1:end-4),'_noise',num2str(scale));
    save(end_name,'noisePACBED','-v7');
end

function noisy = addNoise(orig,scale)
    %scale peak value to value of scale
    ratio = scale./max(max(orig));
    noisy = poissrnd(ratio.*orig);
end