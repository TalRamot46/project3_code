% Shock Solver:
function [m0,mw,e0,ew,u0,uw,xsi,z,utilda,ufront,t,x] = manager(mat,tau);
% provides a self similar solution, provided a material and a temporal powerlaw tau.
% m=mO*T*(mw(2))*tA(mw(3)), in g/cmA2
% e = eO*TA(ew(2))*t*(ew(3)), in J/cmA2
% the same for u in cm/s
% % Xsi - is the self similar front coordinate.
% z - is the dimensionless energy.
% utilda - self similar front velocity
% t - is the coordinate xi
% X - X(:,l) is the self similar volume
% x - x(:,2) is the self similar pressure
% X - x(:,3) is the self similar velocity
% Solve the power laws
% Xsi goes like:
w1=-1/2;
w2=1/2;
w3=-1-tau/2;
% m goes like:
mw(1)=-w1; %PO
mw(2)=-w2; %mat.VO
mw(3)=-w3; %t
% e0 goes like:
ew(1)=3/2;
ew(2)=1/2;
ew(3)=-2+(2+tau)*3/2;
% u goes like:
uw(1)=1/2;
uw(2)=1/2;
uw(3)=tau/2;
% Solve the constants
[t,x]=solve_normalize3(tau,mat.r,20,4);
z=-trapz(t,(x(:,1).*x(:,2)))/mat.r;
z=z-0.5*trapz(t,(x(:,3).^2));
xsi=max(t);
Vtilda=x(1,1);
Ptilda=x(1,2);
utilda=x(1,3);
ufront=utilda*(mat.V0^uw(2));
m0=xsi*(mat.V0^mw(2)); %g/cm*2
e0=z*(mat.V0^ew(2)); %J/cm*2
u0=utilda*(mat.V0^uw(2)); %cm/s
end