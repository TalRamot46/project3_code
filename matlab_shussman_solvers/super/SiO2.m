%Au.m:
%Defines properties for the supersonic Au solver.
mat.alpha=3.5;
mat.beta=1.1;
mat.lambda=0.75;
mat.mu=0.1;
mat.rho0=0.05; %g/cc
HeV=1160500;
mat.f=8.8*10^13/((HeV^mat.beta)*(mat.rho0)^mat.mu); %J/g/HeV*b
mat.g=1/(9175*(HeV^mat.alpha)*(mat.rho0)^mat.lambda); %g/cmA2/HeVAa
mat.sigma=5.670373*10^(-5); %Watt/cm*2/K*4
% mat.sigma=mat.sigma*(1160400)*4*(10^-9); %J/ns/cmA2/HeVA4
%Define units of the conserved quantity, and derive / define tau.
% etta1=0;
% etta2=2;
% etta3=-2;
% tau=(-2/mat.beta)+(((etta1/mat.beta)*(2*mat.alpha+8-3*mat.beta))-2*etta3)/(mat.beta*etta2+(mat.alpha+mat.beta+4)*etta1);
