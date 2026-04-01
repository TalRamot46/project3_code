Au;
[m0,mw,e0,ew,u0,uw,xsi,z,utilda,ufront,t,x] = manager(mat,Pw(3));

t = flipud(t);
x = flipud(x);

times=[0.1];

for i=1:length(times)
m_shock(i,:)=m0*P0^mw(1)*times(i)^mw(3).*t'/xsi;
P_shock(i,:)=P0*times(i)^Pw(3).*x(:,2);
u_shock(i,:)=u0*P0^uw(1)*times(i)^uw(3).*x(:,3)/utilda;
rho_shock(i,:)=1./(mat.V0*x(:,1));
T_shock(i ,:)= (1e12*P_shock(i,:)/ mat.r/ mat.f.*rho_shock(i,:).^(mat.mu-1)).^(1/mat. beta) / 11605;
end

plot(m_shock(1,:), P_shock(1,:), 'r-')