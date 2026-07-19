my_Au;
P0=2.71e12; % dyn/cm^2
tau = -0.45
Pw(3) = tau
[m0,mw,e0,ew,u0,uw,xsi,z,utilda,ufront,t,x] = my_manager(mat,Pw(3));

t = flipud(t);
x = flipud(x);

times=[1e-9];

for i=1:length(times)
m_shock(i,:)=m0*P0^mw(1)*times(i)^mw(3).*t'/xsi;
P_shock(i,:)=P0*times(i)^Pw(3).*x(:,2);
u_shock(i,:)=ufront*(P0)^uw(1)*times(i)^uw(3).*x(:,3)/utilda;
rho_shock(i,:)=1./(mat.V0*x(:,1));
T_shock(i ,:)= 1e4*(P0.*x(: ,2)' .*(times(i))^Pw(3)/ mat.r/ mat.f.*rho_shock(i,:).^(mat.mu-1)).^(1/mat. beta)/11605;

dPdm_numeric_shock(i,:) = diff(P_shock(i,:))./diff(m_shock(i,:));
drhodm_numeric_shock(i,:) = diff(rho_shock(i,:))./diff(m_shock(i,:));
dTdm_numeric_shock(i,:) = diff(T_shock(i,:))./diff(m_shock(i,:));
dPdx_shock(i,:)=dPdm_numeric_shock(i,:).*mid(rho_shock(i,:));
drhodx_shock(i,:)=drhodm_numeric_shock(i,:).*mid(rho_shock(i,:));
dTdx_shock(i,:)=dTdm_numeric_shock(i,:).*mid(rho_shock(i,:));

end

plot(m_shock(1,:), T_shock(1,:))