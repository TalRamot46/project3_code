%Au.m:
%Defines properties for the supersonic Au solver.
mat.alpha=1.5;
mat.beta=1.6;
mat.lambda=0.2;
mat.mu=0.14;
% mat.rho0=19.32; %g/cc
mat.rho0=1.0; %g/cc
HeV=1160500;
mat.f=3.4*10^13/((HeV^mat.beta)*(mat.rho0)^mat.mu); %J/g/HeV*b
mat.g=1/(7200*(HeV^mat.alpha)*(mat.rho0)^mat.lambda); %g/cmA2/HeVAa
mat.sigma=5.670373*10^(-5); %Watt/cm*2/K*4

