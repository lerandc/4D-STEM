function f_addPoissonNoise(file_list,scale_factors)

    for f_name = file_list
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
    end_name = strcat(base_name(1:end-4),'_noise',num2str(scale/1e4));
    save(end_name,'noisePACBED','-v7');
end

function noisy = addNoise(orig,scale)
    noisy = poissrnd(scale.*orig)./scale;
end