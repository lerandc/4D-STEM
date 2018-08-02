%%Crops zero padding from image
%%Author: Luis Rangel DaCosta, lerandc@umich.edu
%%Last comment date: 8-2-2018

function crop = imageCrop(img)
%removes zero padding from masked image
    %cut left
    iter = 1;
    while(~any(img(:,iter)))
        iter = iter+1;
    end
    img = img(:,iter:end);
    
    %cut right
    iter = size(img,2);
    while(~any(img(:,iter)))
        iter = iter-1;
    end
    img = img(:,1:iter);
    
    %cut top
    iter = 1;
    while(~any(img(iter,:)))
        iter = iter+1;
    end
    img = img(iter:end,:);
    
    %cut bottom
    iter = size(img,1);
    while(~any(img(iter,:)))
       iter = iter-1; 
    end
    img = img(1:iter,:);
    
    crop = img;
end
