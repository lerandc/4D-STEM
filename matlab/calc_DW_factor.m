%%Quick script to see some Debye Waller factors, this never worked
%%Author: Luis Rangel DaCosta, lerandc@umich.edu
%%Last comment date: 8-2-2018


%from "An investigation of the Debye-Waller factor and
% Debye temperature of aluminium using nearest
% neighbour central force pair interactions"
%R C G Killean, 1974 J Phyus F Met Phys 4 1908
clearvars

Dt = 340; %debye temperature
temp = 298; %temperature of conditions
M = 1.055e-25; %mass of Cu atom
k = 1.38e-23; %boltzmann constant
hbar = 1.0545718e-34; %hbar

x = Dt/temp;
int_fun = @(y) y./(exp(y)-1); 
fx = @(x) (1./x)*integral(int_fun,0,x);


B = ((24*pi*pi*hbar*hbar*temp)./(M*k*Dt*Dt))*(fx(x)+x/4); 
%this is in angstrom squared
B = B./(1e-20);