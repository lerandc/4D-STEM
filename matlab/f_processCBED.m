%%Processing script to create matlab binary files of raw MRC data from prismatic
%%Author: Luis Rangel DaCosta, lerandc@umich.edu
%%Last comment date: 8-2-2018 as

function f_processCBED(cell_dim,real_pixel,E0,FP,behavior,q_cut,q_cut_style,algorithm,interpolation)
%cell dim is 2 element vector with dimension of simulation cell X, Y
%real pixel is size of real pixel potential sampling in prismatic
%E0 is energy of incident electron in volts
%FP is number of frozen phonons
%q_cut is distance in mrad to cut off the the CBED (reduces output size)
%q_cut_style is 'circ' or 'rect', where circ is a simple distance cut off, and 'rect'
%fits a box to the limits in kx and ky
%algorithm is 'p' for PRISM and 'm' for multislice
%if algorithm == 'p', an interpolation factor must be specified
%behavior is either 'average' or 'unique' for averaging all FP configurations or not

    if algorithm == 'p' && nargin < 8
        error('Not enough input arguments given for PRISM algorithm. Did you remember the interpolation factor?')
    end

    checks = ~[isnumeric(cell_dim) isnumeric(real_pixel) isnumeric(FP)...
                isnumeric(q_cut) ischar(q_cut_style) ischar(algorithm) isnumeric(interpolation)];

    if any(checks)
        error('Check inputs.')
    end
    
    if algorithm == 'p'
        [qxa,qya,qdist] = f_get_PRISM_coords(cell_dim,real_pixel,interpolation);
        f = interpolation;
    elseif algorithm == 'm'
        [qxa,qya,qdist,qmask] = f_get_multislice_coords(cell_dim,real_pixel);
        f = 1;
    else
        error('Check value of algorithm input, value should be ''p'' or ''m''')
    end

    file_list = dir('*FP*.mrc');
    lambda = getLambda(E0);
    
    if q_cut_style == 'circ'
        qdist = qdist.*lambda.*1e3; %convert to mrad
        if algorithm == 'm'
            qdist = imagecrop(qdist.*qmask);
        end
        dist_mask = qdist < q_cut;
    elseif q_cut_style == 'rect'
        qxa = qxa.*lambda.*1e3; %convert to mrad
        qya = qya.*lambda.*1e3;
        if algorithm == 'm'
            qxa = imageCrop(qxa.*qmask);
            qya = imageCrop(qya.*qmask);
        end
        qx_check = abs(qxa) < q_cut;
        qy_check = abs(qya) < q_cut;
        dist_mask = (qx_check+qy_check) == 2;
    else
        error('Invalid masking style given in q_cut_style, value should be ''circ'' or ''rect''')
    end

    if FP < 1
        error('FP should be at least 1')
    elseif FP == 1
        parfor i = 1:length(file_list)
            fname = file_list(i).name;
            map = mrcReader(fname);
            map = rot90((map.stack),2); %rotate to get pointing inwards to atoms
            out_map = imageCrop((dist_mask.*map)).*(f^4); %until prismatic update pushed
            parsave(strcat(fname(1:end-4),'.mat'),out_map);
        end
    else
        if behavior == 'average'
            averageScheme(file_list,dist_mask); % at bottom for clarity's sake
        elseif behavior == 'unique'
            parfor i = 1:length(file_list)
                fname = file_list(i).name;
                map = mrcReader(fname);
                map = rot90(fftshift(map.stack),2); %rotate to get pointing inwards to atoms
                out_map = imageCrop((dist_mask.*map)).*(f^4); %until prismatic update pushed
                parsave(strcat(fname(1:end-4),'.mat'),out_map);
            end
        else
            error('Invalid behavior given for FP configurations, value should be ''average'' or ''unique''')
        end

    end

end


function lambda = getLambda(E0)
    m = 9.19383e-31;
    e = 1.602177e-19;
    c = 299792458;
    h = 6.62607e-34;
    lambda = 1e10*(h/(sqrt(2*m*e*E0)))/(sqrt(1+(e*E0)/(2*m*c*c)));
end

function avg = getAvgFP(f_name,FP,im_size,f)
    avg = zeros(im_size);
    for i = 1:FP
       map = mrcReader(strcat(f_name,num2str(FP),'.mrc'));
       avg = avg+map.stack;
    end
    avg = fftshift(avg./FP).*(f^4);
end

function averageScheme(file_list,dist_mask)
    file_cell = struct2cell(file_list)'; file_cell = file_cell(:,1);
    sort_files = regexp(file_cell,'.*_FP','match','once');
    sort_files = unique(sort_files);
    sort_files = sort_files(~cellfun('isempty',sort_files));
    check_for_4D = contains(sort_files,'_X');
    base_names = sort_files(check_for_4D);
    
    test_img = mrcReader(strcat(base_names{1},'1.mrc'));
    test_img = rotate90(fftshift(test_img.stack),2);
    im_size = size(test_img);
    
    for i = 1:length(base_names)
        result = getAvgFP(base_names{i},n_FP,im_size,f);
        out_map = imageCrop((dist_mask.*result));
        save(strcat(base_names{i},'_avg.mat'),'out_map','-v7');
    end

end

function parsave(name,var)
    save(name,'var','-v7');
end