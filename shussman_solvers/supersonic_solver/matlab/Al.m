%Au.m:
%Defines properties for the supersonic Au solver.

% % Tomer's data:
% mat.alpha=3.32;
% mat.beta=1.2;
% mat.lambda=0.33;
% mat.mu=0;
% % mat.rhoO=19.32; %g/cc
% HeV=1160500;
% mat.f=3.6*10^11*(100^mat.beta)/(HeV^mat.beta); %J/g/HeV*b
% mat.g=1/(7714*(HeV^mat.alpha)); %g/cmA2/HeVAa
% mat.sigma=5.670373*10^(-5); %Watt/cm*2/K*4
% % mat.sigma=mat.sigma*(1160400)^4; %J/ns/cmA2/HeVA4
% mat.r=0.3;

% % Basko's data:
% mat.alpha=3.8;
% mat.beta=1.145;
% mat.lambda=0.5;
% mat.mu=0.063;
% % mat.rhoO=19.32; %g/cc
% HeV=1160500;
% mat.f=12.5*10^14*(0.1^mat.beta)/(HeV^mat.beta); %J/g/HeV*b
% mat.g=5*(0.1)^mat.alpha/(HeV^mat.alpha); %g/cmA2/HeVAa
% mat.sigma=5.670373*10^(-5); %Watt/cm*2/K*4
% % mat.sigma=mat.sigma*(1160400)^4; %J/ns/cmA2/HeVA4
% mat.r=mat.mu/(mat.beta-1)

% Yair's data:
mat.alpha=3.1;
mat.beta=1.2;
mat.lambda=0.3685;
mat.mu=0;
mat.rho0=2.78; %g/cc
HeV=1160500;
mat.f=3.6*10^11*(100^mat.beta)/(HeV^mat.beta)*((mat.rho0)^mat.mu); %J/g/HeV*b
mat.g=1/(1487*(HeV^mat.alpha)*(mat.rho0)^mat.lambda); %g/cmA2/HeVAa
mat.sigma=5.670373*10^(-5); %Watt/cm*2/K*4
% mat.sigma=mat.sigma*(1160400)^4; %J/ns/cmA2/HeVA4
mat.r=0.3;

% % Yair's data B:
% mat.alpha=3.1;
% mat.beta=1.145;
% mat.lambda=0.3685;
% mat.mu=0.063;
% % mat.rhoO=19.32; %g/cc
% HeV=1160500;
% mat.f=12.5*10^14*(0.1^mat.beta)/(HeV^mat.beta); %J/g/HeV*b
% mat.g=1/(3.5*1487*(HeV^mat.alpha)); %g/cmA2/HeVAa
% mat.sigma=5.670373*10^(-5); %Watt/cm*2/K*4
% % mat.sigma=mat.sigma*(1160400)^4; %J/ns/cmA2/HeVA4
% mat.r=mat.mu/(mat.beta-1);

% % Hilik's data:
% mat.alpha=3.1;
% mat.beta=1.2;
% mat.lambda=0.3685;
% mat.mu=0;
% % mat.rhoO=19.32; %g/cc
% HeV=1160500;
% mat.f=3.6*10^11*(100^mat.beta)/(HeV^mat.beta); %J/g/HeV*b
% mat.g=1/(2200*(HeV^mat.alpha)); %g/cmA2/HeVAa
% mat.sigma=5.670373*10^(-5); %Watt/cm*2/K*4
% % mat.sigma=mat.sigma*(1160400)^4; %J/ns/cmA2/HeVA4
% mat.r=0.3;



%Define units of the conserved quantity, and derive / define tau.
%Default is constant temperature:
% etta1=mat.mu/mat.beta;
% etta2=(2-3*mat.mu)/mat.beta;
% etta3=-2/mat.beta;
% tau = solve_for_tau(etta1,etta2,etta3,mat);
