%manager.m:
function [m0,mw,e0,ew,P0,Pw,V0,Vw,u0,uw,xsi,z,Ptilda,utilda,B,t,x] = my_manager(mat,tau)
%manager(mat,tau) provides a self similar solution, provided a material and a temporal
%power law tau.
%m=mO*TA(mw(2))*tA(mw(3)), in g/cmA2
%e = eO*TA(ew(2))*tA(ew(3)), in J/cmAZ
%the same for P in MBar
%xsi - is the self similar front coordinate.
%z - is the dimensionless energy.
%B - is the parameter defined in the article.
%t - is the coordinate xi
%x ~ x(:,l) is the self similar volume and x(:,2) is the volume derivative
%X - x(:,3) is the self similar pressure and X(:,4) is the pressure derivative
% x ~ X( ,5) is the self similar velocity. the temprature is calculated
% via: (X(:,3).*(x(:,l).???(1-mat.mu))).A(l/mat.beta))

%Solve the power laws
%xsi goes like:
tempW=4+2*mat.lambda-4*mat.mu;
w1=(mat.mu-2)/tempW; %A for constant
w2=(-8-2*mat.alpha+2*mat.beta-mat.beta*mat.lambda+(4+mat.alpha)*mat.mu)/tempW; %TO
w3=(-2-2*(4+mat.alpha-mat.beta)*tau+mat.mu*(3+(4+mat.alpha)*tau)-mat.lambda*(2+mat.beta*tau))/tempW; %t
%m goes like:
mw(1)=-w1; %A for constant
mw(2)=-w2; %TO
mw(3)=-w3; %t
%eo goes like:
ew(1)=(2-3*mat.mu)/tempW;
ew(2)=(8+2*mat.alpha+2*mat.beta+3*mat.beta*mat.lambda-3*mat.mu*(4+mat.alpha))/tempW;
ew(3)=(2+2*(4+mat.alpha+mat.beta)*tau+mat.mu*(-1-3*(4+mat.alpha)*tau)+mat.lambda*(2+3*mat.beta*tau))/tempW;
%P goes like:
Pw(1)=(1-mat.mu)*2/tempW;
Pw(2)=(4+mat.alpha+mat.beta*mat.lambda-(4+mat.alpha)*mat.mu)*2/tempW;
Pw(3)=(-1+mat.mu+(4+mat.alpha+mat.lambda*mat.beta)*tau-(4+mat.alpha)*mat.mu*tau)*2/tempW;
%V goes like:
Vw(1)=(-1)*2/tempW;
Vw(2)=(-4-mat.alpha+2*mat.beta)*2/tempW;
Vw(3)=(1-(4+mat.alpha-2*mat.beta)*tau)*2/tempW;
%u goes like:
uw(1)=(-mat.mu)/tempW;
uw(2)=(mat.beta*(2+mat.lambda)-(4+mat.alpha)*mat.mu)/tempW;
uw(3)=(mat.mu+mat.beta*(2+mat.lambda)*tau-(4+mat.alpha)*mat.mu*tau)/tempW;

%Solve the constants
B=(16*mat.sigma*mat.g)/(mat.beta)/3;
B=B*((mat.r*mat.f)^(-(4+mat.alpha)/mat.beta));
[t,x]=solve_normalize(mat.alpha,mat.beta,mat.lambda,mat.mu,mat.r,tau,3000,1,4);
z=-trapz(t,(x(:,1).*x(:,3)))/mat.r;
z=z-0.5*trapz(t,(x(:,5).^2));
xsi=max(t);
Ptilda=x(1,3);
utilda=x(end,5);
P0 = Ptilda * (B^Pw(1)) * ((mat.r*mat.f)^(Pw(2)/mat.beta));     % dyn/cm^2  (or /1e12 to Mbar)
m0 = xsi    * (B^mw(1)) * ((mat.r*mat.f)^(mw(2)/mat.beta));     % g/cm^2
V0 =         (B^Vw(1)) * ((mat.r*mat.f)^(Vw(2)/mat.beta));      % cm^3/g  (since V is specific volume)
u0 = utilda * (B^uw(1)) * ((mat.r*mat.f)^(uw(2)/mat.beta));     % cm/s  (check if your x(:,5) is normalized that way)
e0 = z      * (B^ew(1)) * ((mat.r*mat.f)^(ew(2)/mat.beta));     % erg/g  (then /1e7 to J/g if you want)end
