%createParamters
clearvars
options = getOptions;
blank = 'Option prompts will begin. Enter blank if option is not used.';
fprintf(blank); fprintf('\n');
fileName = input('Enter parameter file name (.txt): ','s');
fID = fopen(fileName,'w');

for iter = 1:size(options,1)
    entry = input(options{iter,1},'s');
    if ~isempty(entry)
        options{iter,3} = entry;
        fprintf(fID,'%s%s%s \n',options{iter,2},':',options{iter,3});
    end
end

fclose(fID);

function opt = getOptions

opt = {'Enter interpolation factor: ','-f',[];
    'Enter interpolation factor in x: ','-fx',[];
    'Enter interpolation factor in y: ','-fy',[];
    'Number of CPU threads to use: ','-j',[];
    'Number of CUDA streams per GPU: ','-S',[];
    'Number of GPUs to use: ','-g',[];
    'Number of simultaneous probes: ','-b',[];
    'Number of simultaneous probes for CPU workers: ','-bc',[];
    'Number of simultaneous probes for GPU workers: ','-bg',[];
    'Slice thickness (Angstroms): ','-s',[];
    'Pixel size (Angstroms): ','-p',[];
    'Detector angle step size (mrad): ','-d',[];
    'Cell dimension (Angstroms, x y z):','-c',[];
    'Tile unit cell x y z number of times in x, y, z: ','-t',[];
    'Algorithm (p or m for prism or multislice):','-a','';
    'Energy of electron beam (keV): ','-E',[];
    'Maximum probe angle (mrad): ','-A',[];
    'Potential bound (Angstroms): ','-P',[];
    'Do CPU work (1/0): ','-C',[];
    'Do streaming mode (1/0): ','-streaming-mode',[];
    'Probe step size (Angstroms): ','-r',[];
    'Probe step size in X (Angstroms): ','-rx',[];
    'Probe step size in Y (Angstroms): ','-ry',[];
    'Random seed (int): ','-rs',[];
    'Probe X tilt (mrad): ','-tx',[];
    'Probe Y tilt (mrad): ','-ty',[];
    'Probe defocus (Angstroms): ','-df',[];
    'Maximum probe semiangle (mrad): ','-sa',[];
    'Scan window X (min max, fractional coords.): ','-wx',[];
    'Scan window Y (min max, fractional coords.): ','-wy',[];
    'Number of frozen phonon configurations: ','-F',[];
    'Include Debye-Waller factors (1/0): ','-te',[];
    'Consider occupancy values (1/0): ','-oc',[];
    'Save 2D output (ang_min, ang_max in mrad): ','-2D',[];
    'Save 3D output (1/0): ','-3D',[];
    'Save 4D output (1/0): ','-4D',[];
    };
end
