%F.m:
function xp=F(t,x,alpha,beta,tau)
%Calculates the derivatives, used in ode45, for the numerical
%integration of the self similar profile
w3=-0.5+0.5*(beta-alpha-4)*tau;
xp=zeros(2,1);
xp(1)=x(2);
xp(2)=(x(1)^(beta-alpha-4))*[w3*t*x(2)+tau*x(1)]-(alpha+3)*(1/x(1))*(x(2)^2);
end
