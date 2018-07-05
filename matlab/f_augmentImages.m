function f_augmentImages(file_list,translate_range,scale_range,shear_range,rotate_range)

end

function new = translate(orig, trans)
%trans is 2 element vector with x displacment in pos 1 and y displacement
%in pos 2
    trans_mat = diag([1 1 1]);
    trans_mat(3,1:2) = trans;
    
    tform = affine2d(trans_mat);
    
    new = imwarp(orig,tform);
    new = imresize(new,size(orig),'lanczos3');
end

function new = scale(orig,trans)
    trans_mat = diag([trans(1) trans(2) 1]);
    
    tform = affine2d(trans_mat);
    
    buffer = imwarp(orig,tform);
    new = zeros(size(orig));
    offset_x = floor((size(orig,2)-size(buffer,2))/2);
    offset_y = floor((size(orig,1)-size(buffer,1))/2);
    new(offset_y:offset_y+size(buffer,2)-1,offset_x:offset_x+size(buffer,2)-1) = ...
        buffer;
end

function new = shear(orig,trans)
    trans_mat = diag([1 1 1]);
    trans_mat(1,2) = trans(2);
    trans_mat(2,1) = trans(1);
    
    tform = affine2d(trans_mat);
    
    new = imwarp(orig,tform);
    new = imresize(new,size(orig),'lanczos3');
end

function new = rotate(orig, trans)
    trans_mat = [cos(trans) sin(trans) 0;
                -sin(trans) cos(trans) 0;
                0 0 1];
    
    tform = affine2d(trans_mat);
    
    new = imwarp(orig,tform);
    new = imresize(new,size(orig),'lanczos3');
end
