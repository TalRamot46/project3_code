%Be.m:
%Defines properties for the supersonic Au solver.
mat.alpha=4.893;
mat.beta=1.0902;
mat.lambda=0.6726;
mat.mu=0.0701;
% mat.rho0=1.85; %g/cc
mat.rho0=1; %g/cc
HeV=1160500;
mat.f=8.8053*10^13/((HeV^mat.beta)*(mat.rho0)^mat.mu); %J/g/HeV*b
mat.g=1/(402.8102*(HeV^mat.alpha)*(mat.rho0)^mat.lambda); %g/cmA2/HeVAa
mat.sigma=5.670373*10^(-5); %Watt/cm*2/K*4
% mat.sigma=mat.sigma*(1160400)*4*(10^-9); %J/ns/cmA2/HeVA4
mat.r=0.5529;
%Define units of the conserved quantity, and derive / define tau.
%Default is constant temperature:
% etta1=mat.mu/mat.beta;
% etta2=(2-3*mat.mu)/mat.beta;
% etta3=-2/mat.beta;
% tau = solve_for_tau(etta1,etta2,etta3,mat);
