%solve_for_tau.m:
function tau = solve_for_tau(etta1,etta2,etta3,mat)
%Given a material mat, and a conserved quantitiy units etta,
%solves for the temporal evolution of the temperature
%Define temporary constants
A=zeros(2);
B=zeros(2,1);
%Fill A,B
A(1,1)=2*mat.beta+mat.lambda*mat.beta-mat.mu*(4+mat.alpha);
A(1,2)=mat.mu;
A(2,1)=3*mat.mu*(4+mat.alpha)-2*mat.alpha-2*mat.beta-3*mat.lambda*mat.beta-8;
A(2,2)=2-3*mat.mu;
B(1)=-mat.beta*etta1;
B(2)=-mat.beta*etta2;
%solve
x=linsolve(A,B);
tau=x(1)*(8-3*mat.beta+2*mat.alpha)+mat.beta*etta3;tau=tau/x(2);
tau=(tau-2)/mat.beta;
end
