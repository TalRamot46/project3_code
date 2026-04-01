%F.m:
function xp=F(t,x,alpha,beta,lambda,mu,r,tau)
%Calculates the derivatives, used in ode45, for the numerical
%integration of the self similar profile
mechane=4+2*lambda-4*mu;
wm3=2+2*(4+alpha-beta)*tau-mu*(3+(4+alpha)*tau)+lambda*(2+beta*tau);
wm3=wm3/mechane;
wu3=mu+beta*(2+lambda)*tau-(4+alpha)*mu*tau;
wu3=wu3/mechane;
wP3=-1+mu+(4+alpha+beta*lambda)*tau-(4+alpha)*mu*tau;
wP3=wP3*2/mechane;
wV3=1-(4+alpha-2*beta)*tau; wV3=wV3*2/mechane;
w3=-wm3;
temp3=(4+alpha)/beta;
temp3=1;
ezerA=((wV3+wP3)*x(1)*x(3)+w3*t*(x(1)*x(4)+x(2)*x(3)))/r;
ezerA2=x(3)*(wV3*x(1)+w3*t*x(2));
temp2=(1-mu)*x(3)*(x(1)^-mu)*x(2)+(x(1)^(1-mu))*x(4);
ezerB=temp2*lambda*(x(1)^(lambda-1))*x(2);
temp=x(3)*(x(1)^(1-mu));
ezerB=ezerB*(temp^((4+alpha-beta)/beta));

ezerC=temp2^2; ezerC=ezerC*(temp^((4+alpha-2*beta)/beta));
ezerC=ezerC*(x(1)^lambda);ezerC=ezerC*(4+alpha-beta)/beta;

ezerD=(2*(1-mu)*(x(1)^(-mu))*x(2)*x(4)-mu*(1-mu)*x(3)*(x(1)^(-mu-1))*(x(2)^2));

temp4=(x(1)^lambda); temp4=temp4*(temp^((4+alpha-beta)/beta));
ezerE=(wu3+w3)*(wV3*x(1)+w3*t*x(2));ezerE=ezerE+w3*t*(wV3*x(2)+w3*x(2));
ezerE=-ezerE;ezerForP=ezerE;ezerE=ezerE*(x(1)^(1-mu));

ezerF=(1-mu)*x(3)*(x(1)^-mu);
ezerF=ezerF-(x(1)^(1-mu))*(w3^2)*(t^2);

xp=zeros(5,1);

xp(5)=wV3*x(1)+w3*t*x(2); %u'=

xp(1)=x(2); %V'

xp(3)=x(4); %P'

xp(2)=(ezerA+ezerA2)/temp3;xp(2)=(xp(2)-ezerB-ezerC)/temp4;xp(2)=(xp(2)-ezerD-ezerE)/ezerF; %V"

xp(4)=-(w3^2)*(t^2)*xp(2)+ezerForP; %P"
end
