function f_effSourceSize(source_size,pixel_size,base_name,base_ext,array_size)
    img_map = getPrismMap(array_size(1)-1,array_size(2)-1);

    imsize = size(loadImageFromMat(strcat(base_name,'_X0_Y0',base_ext)));
   
    sig = (source_size/pixel_size)./(2.355);
    gaussKernel = fpsecial('gaussian',array_size,sig);

    orig_CBED_array = get4DArray(base_name,base_ext,img_map,imsize);

    result = convolve2D(orig_CBED_array,gaussKernel);
    algined_result = alignResult(result,img_map);

    save4DArray(algined_result,base_name,base_ext,source_size);

    
end

function img = loadImageFromMat(f_name)
    data = load(f_name);
    fields = fieldnames(data);
    img = rot90(data.(fields{1}),2);
end

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

function array = get4DArray(base_name,base_ext,map,imsize)
    %create empty array of X x Y x kx x ky 
    array = zeros(cat(2,size(map),imsize));
    for i = 1:size(map,1)
        for j = 1:size(map,2)
            ID = map{i,j};
            fname = strcat(base_name,'_X',num2str(ID(1)),...
                '_Y',num2str(ID(2)),base_ext);
            array(i,j,:,:) = (loadImageFromMat(fname));
        end
    end

end

function save4DArray(result,base_name,base_ext,source_size)
    
    mid_string = strcat('_blur',num2str(source_size));
    
    for i = 0:size(result,1)-1
        for j = 0:size(result,2)-1
            f_name = strcat(base_name,mid_string,'_X',num2str(j),...
                '_Y',num2str(i),base_ext);
            cbed = result(i+1,j+1,:,:);
            save(f_name,'cbed')
        end
    end

end

function result = convolve2D(CBED_array,kernel)

    f_kernel = fft2(kernel);
    result = CBED_array;
    
    for k = 1:size(CBED_array,3)
        for l = 1:size(CBED_array,4)                           
            result(:,:,k,l) = (ifft2(f_kernel.*((fft2(CBED_array(:,:,k,l))))));
        end
    end

end

function aligned = alignResult(CBED_array,map)

    new_map = fftshift(map);
    aligned = zeros(size(CBED_array));
    for i = 1:size(map,1)
        for j = 1:size(map,2)
            ID = new_map{i,j};
            aligned(i,j,:,:) = CBED_array(ID(2)+1,ID(1)+1,:,:);
        end
    end

end
