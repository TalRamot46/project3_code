% solve_norma1ize3.m:
function [t,x]=solve_normalize3(tau,r,iternum,xsi0)
% Normalizes the solution using a binary shooting method. if thesolution
% for P(O) is larger than 1, decrease xsi_f. otherwise, increase.
% xsiO - the initial guess. it is advised to begin from xsiO which is
% power of 2, so the solution will asymptotically reach any value
% between 2*Xsio and O.
% iternum - the number of iterations.
first_change=floor(log(xsi0)/log(2));
wm3=1+(1/2)*tau;
x0=zeros(3,1);
a=zeros(20);a(1)=xsi0;
for i=1:20
% hugoniot
x0(1) = r/(r+2);
x0(3) = wm3*a(i)*2/r*x0(1);
x0(2) = wm3*a(i)*x0(3);
% solve
options = odeset('AbsTol',1e-9,'RelTol',1e-9);
[t,x]=ode45(@(t,x) F3(t,x,tau,r) ,[a(i),0.0001],x0,options);
if(x(end,2)>1)
a(i+1)=a(i)-2^(-i+first_change);
else
a(i+1)=a(i)+2^(-i+first_change);
end
end
end