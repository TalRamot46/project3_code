close all;
clear all;
Au;
tau=0.0;
[m0,mw,e0,ew,P0,Pw,V0,Vw,u0,uw,xsi,z,Ptilda,utilda,B,t,x] = manager(mat,tau);
times=[1];

t = flipud(t);
x = flipud(x);

T0=1;
for i=1:length(times)
m_heat(i,:)=m0*T0^mw(2)*times(i)^mw(3).*t'/xsi;
P_heat(i,:)=P0*T0^Pw(2)*times(i)^Pw(3).*x(:,3)/Ptilda;
T_heat(i,:)=100*T0*times(i)^tau*(x(:,3).*(x(:,1).^(1-mat.mu))).^(1/mat.beta);
u_heat(i,:)=u0*T0^uw(2)*times(i)^uw(3).*x(:,5)/utilda;
rho_heat(i,:)=1./(V0*T0^Vw(2)*times(i)^Vw(3).*x(:,1));
end