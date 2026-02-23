Au;
tau=0.141;
[m0,mw,e0,ew,xsi,z,A,t,x] = manager(mat,tau);
times=[1];

% T0=1.9;
T0=2;
for i=1:length(times)
m_heat(i,:)=m0*T0^mw(2)*times(i)^mw(3).*t'/xsi;
x_heat(i,:)=m_heat(i,:)/mat.rho0;
T_heat(i,:)=100*T0*times(i)^tau*(x(:,1));
end

