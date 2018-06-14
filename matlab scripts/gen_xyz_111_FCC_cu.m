clearvars
close all

a = 3.615; %angstroms
b = a; c = a; %cubic

alpha = pi/2; %orthorhombic
beta = alpha; gamma = alpha;

pos = [0 0 0; 0 1/2 1/2; 1/2 0 1/2; 1/2 1/2 0]; %wyckoff positions, each layer is different atom type

num_rep_x = 50;
num_rep_y = 50;
num_rep_z = 50;
num_rep = 100;
num_pos = size(pos,1);

coords = zeros(num_rep_x*num_rep_y*num_rep_z*num_pos,3);
count = 1;
%only works for orthorhombic cells
for i = 1:num_rep_x
for j = 1:num_rep_y
for k = 1:num_rep_z
    for current_pos = 1:num_pos
        coords(count,[1 2 3]) = pos(current_pos,:).*[a b c]+[a*(i-1) b*(j-1) c*(k-1)];
        count = count+1;
    end
end
end
end

%end goal is to get (111) plane normal to +z
%following loosely: https://sites.google.com/site/glennmurray/Home/rotation-matrices-and-formulas/rotation-about-an-arbitrary-axis-in-3-dimensions

%rotating vector is from (a,0,0) to (0,b,0)
vec = -[a 0 0]+[0 b 0];

%translate so origin is in center
center = mean(coords);
coords = coords-center;
plot3(coords(1:1000,1),coords(1:1000,2),coords(1:1000,3),'o')
hold on

%rotate coords until rotating vector is in xz plane
check = dot(vec,[0 1 0]);
if check
    theta = 0.1;
    while(abs(check)>1e-3)
        check = dot(rotZ(theta,vec),[0 1 0]);
        theta = theta+0.0001;
    end
end
z_theta = theta;
coords = rotZ(z_theta,coords);
vec2 = rotZ(z_theta,vec);
plot3(coords(1:1000,1),coords(1:1000,2),coords(1:1000,3),'o')

%rotate coords until rotating vector is parallel to Z
check = 1-dot(vec2,[0 0 1]);
if check
    theta = 0.1;
    while(abs(check(1))>1e-3 || abs(check(2)) > 1e-3)
        check = [0 0 1]-(rotY(theta,vec2));
        theta = theta+0.00001;
    end
end
y_theta = theta;
coords = rotY(y_theta,coords);
vec3 = rotY(y_theta,vec2);
plot3(coords(1:1000,1),coords(1:1000,2),coords(1:1000,3),'o')

%rotate desired angle around Z
coords = rotZ(-atan(1/sqrt(2)),coords);

%invert steps in order
coords = rotY(-y_theta,coords);
coords = rotZ(-z_theta,coords);

%could translate back, but convenient to have it centered
%cut into a rectangular block

coords = rotZ((-45*pi/180),coords);
coords = rotX(pi/2,coords);
coords(abs(coords(:,1))>30,1) = NaN;
coords(abs(coords(:,2))>30,2) = NaN;
coords(abs(coords(:,3))>30,3) = NaN;

check_list = ~isnan(sum(coords,2));
final_coords = coords(check_list,:);
figure
plot3(final_coords(:,1),final_coords(:,2),final_coords(:,3),'o')

function end_coords = rotZ(theta,coords)
  rot_mat = [cos(theta) -sin(theta) 0;
            sin(theta) cos(theta) 0;
             0 0 1];
         
  end_coords = zeros(size(coords));       
  for i = 1:size(coords,1)
      end_coords(i,:) = rot_mat*(coords(i,:)');
  end
         
end

function end_coords = rotY(theta,coords)
  rot_mat = [cos(theta) 0 sin(theta);
             0 1 0
             -sin(theta) 0 cos(theta)];
         
  end_coords = zeros(size(coords));       
  for i = 1:size(coords,1)
      end_coords(i,:) = rot_mat*(coords(i,:)');
  end
  
end

function end_coords = rotX(theta,coords)
  rot_mat = [1 0 0;
             0 cos(theta) -sin(theta)
             0 sin(theta) cos(theta)];
         
  end_coords = zeros(size(coords));       
  for i = 1:size(coords,1)
      end_coords(i,:) = rot_mat*(coords(i,:)');
  end
  
end