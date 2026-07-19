%Au.m:
%Defines properties for the supersonic Au solver.
mat.alpha=1.5;
mat.beta=1.6;
mat.lambda=0.2;
mat.mu=0.14;
% mat.rhoO=19.32; %g/cc
HeV=1160500;
mat.f=3.4*10^13; %J/g/HeV*b
mat.g=1/7200; %g/cmA2/HeVAa
mat.sigma=5.670373*10^(-5) * HeV^4; %Watt/cm*2/K*4
% mat.sigma=mat.sigma*(1160400)*4*(10^-9); %J/ns/cmA2/HeVA4
mat.r=0.25;
mat.V0=1/19.32;
%Define units of the conserved quantity, and derive / define tau.
%Default is constant temperature:

