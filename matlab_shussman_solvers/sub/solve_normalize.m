%solve_normalize.m:
function [t,x]=solve_normalize(alpha,beta,lambda,mu,r,tau,iternum,xsi0,PO)
%Normalizes the solution using a binary shooting method. for each xsi_f,
%if P(O) > 0 reduce P(l), else (if the integration stops) add to P(1).
%xsi_f is later determined via normalization of the equations, but can
%also be determined using a double shooting method.
%xsiO,PO - the initial guess. it is advised to begin from Po which is
%power of 2, so the solution will asymptotically reach any value
%between 2*PO and O.

%iternum - the number of iterations.
first_change=floor(log(PO)/log(2));
id='MATLAB:ode45:IntegrationTolNotMet';
warning('off',id);
a=zeros(2);a(1)=xsi0;
b=zeros(iternum);b(1)=4;
for i=1:2
for j=1:iternum
lastwarn('');
% options = odeset('AbsTol',5e-5,'RelTol',5e-3);
% [t,x]=ode45(@(t,x)F(t,x,alpha,beta,lambda,mu,r,tau),[a(i),0],[0.02,0,b(j),0,0]);
[t,x]=ode45(@(t,x)F(t,x,alpha,beta,lambda,mu,r,tau),[a(i),0],[0.002,0,b(j),0,0]);
if(size(lastwarn)>0)
b(j+1)=b(j)+2^(-j+first_change);
else
b(j+1)=b(j)-2^(-j+first_change);
end
end
%if(x(size(x(:,1) ,1) ,3)*(x(size(x(:,1) ,1) ,1)^(1-mu))>1)
%a(i+1)=a(i)-2^(-i);
%else
%a(i+1)=a(i)+2^(-i);
%end
calibrator_helper=(4+alpha-beta)/beta;
calibrator=2*(1-calibrator_helper)/(lambda-mu+(2-mu)*calibrator_helper);
c=(1/(x(size(x(:,1),1),3)*(x(size(x(:,1),1),1)^(1-mu))))^(1/(2*calibrator+2-mu*calibrator));
a(i+1)=c;
end
end
