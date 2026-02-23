%solve_normalize.m:
function [t,x]=solve_normalize(alpha,beta,tau,iternum,xsi0)

%Normalizes the solution using a binary shooting method. if the solution
%for T(O) is larger than 1, decrease xsi_f. otherwise, increase.
%XsiO - the initial guess. it is advised to begin from xsiO which is a
%power of 2, so the solution will asymptotically reach any value
%between 2*xsio and O.
%iternum - the number of iterations.

first_change = floor(log(xsi0)/log(2));
a=zeros(iternum,1);a(1)=xsi0;
for i=1:iternum
    [t,x]=ode45(@(t,x) F(t,x,alpha,beta,tau),[a(i),0],[0.001,-1000]);
    if(x(size(x(:,1),1),1)>1)
        a(i+1)=a(i)-2^(-i+first_change);
    else
        a(i+1)=a(i)+2^(-i+first_change);
    end
%     a(i+1)=(x(size(x(:,1),1),1))^((beta-alpha-4)/2)
    if(abs(x(size(x(:,1),1),1)-1)<1e-5)break;
end
end
