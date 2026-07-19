my_Au;
tau=0;
[m0,mw,e0,ew,xsi,z,A,t,x] = manager(mat,tau);
times=[0.5];

% T0=1.9;
T0=1;
for i=1:length(times)
m_heat(i,:)=m0*T0^mw(2)*times(i)^mw(3).*t'/xsi;
x_heat(i,:)=m_heat(i,:)/mat.rho0;
T_heat(i,:)=T0*times(i)^tau*(x(:,1));
e_heat(i,:)=mat.f*T_heat(i,:).^mat.beta*mat.rho0^(-mat.mu);
m_vec = fliplr(m_heat(i,:));
e_vec = fliplr(e_heat(i,:));
E_rad_prof(i,:) = cumtrapz(m_vec, e_vec)  % <-- cumulative integral
E_rad_total(i)  = E_rad_prof(i,end);
end
a_Hev = 4*mat.sigma/(3e10);
plot(m_vec(1,:), T_heat(1,:))