function f_createPACBED(radii,centers,base_name,base_ext,out_name,out_ext,array_size)
    imsize = size(loadImageFromMat(strcat(base_name,'_X0_Y0',base_ext)));
    prism_map = getPrismMap(array_size(1),array_size(2));
    
    %load once to prevent slow down from tons of file reads
    orig_CBED_array = get4DArray(base_name,base_ext,prism_map,imsize);

    xvec = 1:size(prism_map,2);
    yvec = 1:size(prism_map,1);
    [xmesh, ymesh] = meshgrid(xvec,yvec);

    %prism_map = reshape(prism_map,[],1);
    orig_CBED_array = reshape(orig_CBED_array,cat(2,array_size(1)*array_size(2), imsize));

    for radius = radii
        for center_iter = 1:length(centers)
            center = centers(center_iter,:);
            xmesh_center = xmesh-center(2);
            ymesh_center = ymesh-center(1);
            dist_grid = sqrt(xmesh_center.^2+ymesh_center.^2);

            mask = dist_grid < radius;

            mask = reshape(mask,[],1);
            pacbed = single(squeeze(sum(orig_CBED_array(mask,:,:),1))./(sum(sum(mask))));

            savePACBED(pacbed,out_name,out_ext,center,radius);
        end
    end

end

function img = loadImageFromMat(f_name)
    data = load(f_name);
    fields = fieldnames(data);
    img = data.(fields{1});
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

function savePACBED(pacbed,out_name,out_ext,center,radius)
    %need to establish a convention
    f_name = strcat(out_name,'_',num2str(center(1)),'_',num2str(center(2)),...
        '_',num2str(radius),out_ext);
    save(f_name,'pacbed','-v7');

end