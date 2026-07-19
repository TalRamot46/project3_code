%Au.m:
%Defines properties for the supersonic Pb solver.
mat.alpha=2.02;
mat.beta=1.35;
mat.lambda=0.23;
mat.mu=0.14;
% mat.rho0=10.6; %g/cc
mat.rho0=1.0; %g/cc
HeV=1160500;
mat.f=3.5*10^13/((HeV^mat.beta)*(mat.rho0)^mat.mu); %J/g/HeV*b
mat.g=1/(13333*(HeV^mat.alpha)*(mat.rho0)^mat.lambda); %g/cmA2/HeVAa
mat.sigma=5.670373*10^(-5); %Watt/cm*2/K*4

