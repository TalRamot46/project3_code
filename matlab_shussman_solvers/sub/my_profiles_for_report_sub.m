close all;
clear all;
my_Au;
tau=0.0;
[m0,mw,e0,ew,P0,Pw,V0,Vw,u0,uw,xsi,z,Ptilda,utilda,B,t,x] = my_manager(mat,tau);
times=[1e-9];

t = flipud(t);
x = flipud(x);

% T0=2;
T0=1;
for i=1:length(times)
m_heat(i,:)=m0*T0^mw(2)*times(i)^mw(3).*t'/xsi;
P_heat(i,:)=P0*T0^Pw(2)*times(i)^Pw(3).*x(:,3)/Ptilda;

dPdm_heat(i,:)=P0*T0^Pw(2)*times(i)^Pw(3).*x(:,4)/Ptilda/(m0*T0^mw(2)*times(i)^mw(3)/xsi);

T_heat(i,:)=T0*times(i)^tau*(x(:,3).*(x(:,1).^(1-mat.mu))).^(1/mat.beta);
u_heat(i,:)=u0*T0^uw(2)*times(i)^uw(3).*x(:,5)/utilda;
rho_heat(i,:)=1./(V0*T0^Vw(2)*times(i)^Vw(3).*x(:,1));

drhodm_heat(i,:)=-(V0*T0^Vw(2)*times(i)^Vw(3).*x(:,2)')/(m0*T0^mw(2)*times(i)^mw(3)/xsi).*rho_heat(i,:).^2;
dPdx_heat(i,:)=dPdm_heat(i,:).*rho_heat(i,:);
drhodx_heat(i,:)=drhodm_heat(i,:).*rho_heat(i,:);

zeta=((x(:,3).*(x(:,1).^(1-mat.mu))).^(1/mat.beta)).^(4+mat.alpha);
dzetady=diff(zeta)./diff(t/xsi);

F_heat(i,:)=-mid(x(:,1).^(mat.lambda)).*dzetady*xsi;

dPdm_numeric_heat(i,:) = diff(P_heat(i,:))./diff(m_heat(i,:));
drhodm_numeric_heat(i,:) = diff(rho_heat(i,:))./diff(m_heat(i,:));
dTdm_numeric_heat(i,:) = diff(T_heat(i,:))./diff(m_heat(i,:));
dTdx_heat(i,:)= dTdm_numeric_heat(i,:).*mid(rho_heat(i,:));

end

for i =1:length(times)
hold on;
grid on;
plot(m_heat(i,:),P_heat(i,:))
end
