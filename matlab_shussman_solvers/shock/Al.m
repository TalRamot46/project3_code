%Au.m:
%Defines properties for the supersonic Au solver.
mat.alpha=3.1;
mat.beta=1.2;
mat.lambda=0.3685;
mat.mu=0;
% mat.rhoO=19.32; %g/cc
HeV=1160500;
mat.f=3.6*10^11*(100^mat.beta)/(HeV^mat.beta); %J/g/HeV*b
mat.g=1/(7714*(HeV^mat.alpha)); %g/cmA2/HeVAa
mat.sigma=5.670373*10^(-5); %Watt/cm*2/K*4
mat.sigma=mat.sigma*(1160400)*4*(10^-9); %J/ns/cmA2/HeVA4
mat.r=0.66667;
mat.V0=1/2.78;
%Define units of the conserved quantity, and derive / define tau.
%Default is constant temperature:

