clearvars
close all
figure(1)
clf
tumo = mrcReader
map = tumo.stack;
map = permute(map,[2 3 1]);
map(:,:,3) = log(map(:,:,3));
% bright field image
axes('position',[0.0 0.5 0.5 0.5]); 
imagesc([sum(map(:,:,1:20),3)]); 
axis equal off

% annular bright field image
axes('position',[0.5 0.5 0.5 0.5]); 
imagesc([sum(map(:,:,11:20),3)]); 
axis equal off

colormap(gray(256))
% annular dark field image
axes('position',[0.0 0.0 0.5 0.5]); 
imagesc([sum(map(:,:,21:40),3)]); 
axis equal off

% high angle annular dark field image
axes('position',[0.5 0.0 0.5 0.5]); 
imagesc([sum(map(:,:,61:end),3)]); 
axis equal off

colormap(gray(256))