% F3.m:
function xp=F3(t,x,tau,r)
% Calculates the derivatives, used in ode45, for the numerical
% integration of the self similar profile
wm3=1+(1/2)*tau;
wu3=tau/2;
wP3=tau;
wV3=0;
w3=-wm3;
xp=zeros(3,1);
xp(1)=-(x(1)*(x(2)*tau - x(3)*w3*wu3*t))/(w3*t*(- x(1)*w3^2*t^2+x(2)+x(2)*r));
xp(3)=-(x(1)*(x(2)*tau - x(3)*w3*wu3*t))/(- x(1)*w3^2*t^2+x(2)+x(2)*r);
xp(2)=-(x(2)*(x(3)*wu3+r*x(3)*wu3- x(1)*tau*w3*t))/(- x(1)*w3^2*t^2+x(2)+x(2)*r);
