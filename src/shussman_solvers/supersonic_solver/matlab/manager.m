%Manager.m:
function [m0,mw,e0,ew,xsi,z,A,t,x] = manager(mat,tau)
%provides a self similar solution, provided a material and a temporal power law tau.

%m=mO*T*(mw(2))*t^(mw(3)), in g/cm???2
%e = eo*T*(ew(2))*t*(ew(3)), in J/cm*2
%xsi - is the self similar front coordinate.
%z - is the dimensionless energy.
%A - is the parameter defined in the article.
%t - is the coordinate xi
% x - x(:,l) is the self similar temperature and x(:,2) is the temparture derivative

% Solve the power laws
% Xsi goes like:
w1=0.5; %A for constant
w2=(mat.beta-mat.alpha-4)/2; %T0
w3=-0.5+0.5*(mat.beta-mat.alpha-4)*tau; %t
%m goes like:
mw(1)=-w1; %A for constant
mw(2)=-w2; %TO
mw(3)=-w3; %t
%eO goes like:
ew(1)=mw(1);
ew(2)=mat.beta+mw(2);
ew(3)=mat.beta*tau+mw(3);

% Solve for the constants
A=3*mat.f*mat.beta/16/mat.sigma/mat.g;
[t,x]=solve_normalize(mat.alpha,mat.beta,tau,100,1);
z=-trapz(t,(x(:,1).^mat.beta));
xsi=max(t);
m0=xsi*(A^mw(1))*(10^(-9*(-mw(2)*tau)))*1160500^mw(2)*(1e-9)^mw(3); %g/cmA2
e0=z*mat.f*(A^ew(1))*(10^(-9*(-ew(2)*tau)))*1160500^ew(2)*(1e-9)^ew(3)/1e9/100; %J/cm*2
end
