
load('613_multislice_X0_Y0_FP1.mat');
ref = out_map;
figure; imagesc(ref);

translate_range_x = linspace(-10,10,21); %range in pixels
translate_range_y = linspace(-10,10,21);
scale_range = linspace(0.7,1.3,21); %fractional change;
shear_range = linspace(0,0.1,11);
rotate_range = linspace(-pi/4,pi/4,25);

f = imagesc();

% for i = translate_range_x
%     for j = translate_range_y
%         new = translate(ref,[i j]);
%         f.CData = new;
%         drawnow
%         pause(0.05);
%     end
% end

%%%scaling needs to be fixed whenever real image set is created
for i = scale_range
   new = scale(ref, [i i]); 
        f.CData = new;
        drawnow
        pause(0.2);
end

for i = shear_range
   new = shear(ref, [i i]);
        f.CData = new;
        drawnow
        pause(0.2);
end

for i = rotate_range
   new = rotate(ref,i);
        f.CData = new;
        drawnow
        pause(0.2);
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
