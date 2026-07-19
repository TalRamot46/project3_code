close all;
N=10000;
m_final=zeros(length(times),N);
P_final=zeros(length(times),N);
u_final=zeros(length(times),N);
T_final=zeros(length(times),N);
rho_final=1/mat.V0*ones(length(times),N);
dPdx_final=zeros(length(times),N);
dTdx_final=zeros(length(times),N);
drhodx_final=zeros(length(times),N);

for i=1:length(times)
m_trans= max(m_heat(i,:));
relevant = find(m_shock(i,:)>=m_trans);
relevant = relevant(1:end-1);
relevant_heat=find(rho_heat(i,:)<=rho_shock(i,relevant(1)));
m_final(i,1:length(relevant)+length(relevant_heat) )=([m_heat(i,relevant_heat) m_shock(i,relevant)]);
m_final(i,length(relevant)+length(relevant_heat):N)=max(m_shock(i,:))*linspace(1,1.2,N+1-length(relevant)-length(relevant_heat));
P_final(i,1:length(relevant)+length(relevant_heat)) = ([P_heat(i,relevant_heat) P_shock(i,relevant)]);
u_final(i,1:length(relevant)+length(relevant_heat)) = ([u_heat(i,relevant_heat) u_shock(i,relevant)]);
T_final(i,1:length(relevant)+length(relevant_heat)) = ([T_heat(i ,relevant_heat) T_shock(i,relevant)]);
rho_final(i,1:length(relevant)+length(relevant_heat)) = ([rho_heat(i,relevant_heat) rho_shock(i,relevant)]);
dPdx_final(i,1:length(relevant)+length(relevant_heat)) = ([dPdx_heat(i ,relevant_heat) dPdx_shock(i,relevant)]);
dTdx_final(i,1:length(relevant)+length(relevant_heat)) = ([dTdx_heat(i ,relevant_heat) dTdx_shock(i,relevant)]);
drhodx_final(i,1:length(relevant)+length(relevant_heat)) = ([drhodx_heat(i ,relevant_heat) drhodx_shock(i,relevant)]);
end

T_final = T_final * 11605;
dTdx_final = dTdx_final * 11605;
P_final = P_final * 1e12;
dPdx_final = dPdx_final * 1e12;

for i=1:length(times)
    lambda1(i,:) = (1-mat.mu)*mat.r*mat.f*T_final(i,:).^mat.beta.*rho_final(i,:).^(-mat.mu);
    lambda2(i,:) = mat.beta*mat.r*mat.f*T_final(i,:).^(mat.beta-1).*rho_final(i,:).^(1-mat.mu);
    CV(i,:) = mat.beta*mat.f*T_final(i,:).^(mat.beta-1).*rho_final(i,:).^(-mat.mu);
    As1(i,:) = dPdx_final(i,:)./(rho_final(i,:).*lambda1(i,:)+T_final(i,:)./rho_final(i,:)./CV(i,:).*lambda2(i,:).^2);
    As2(i,:) = - drhodx_final(i,:)./rho_final(i,:);
    As(i,:) = As1(i,:) + As2(i,:);
    At1(i,:) = dPdx_final(i,:)./(rho_final(i,:).*lambda1(i,:));
    At2(i,:) = - lambda2(i,:).*dTdx_final(i,:)./(rho_final(i,:).*lambda1(i,:)) ;
    At3(i,:) = - drhodx_final(i,:)./rho_final(i,:);
    At(i,:) = At1(i,:) + At2(i,:) + At3(i,:);

    size1(i,:) = As(i,:) .* dPdx_final(i,:)./rho_final(i,:);
    size2(i,:) = At(i,:) .* dPdx_final(i,:)./rho_final(i,:);
    size3(i,:) = -drhodx_final(i,:) ./ rho_final(i,:) .* dPdx_final(i,:) ./ rho_final(i,:);

end

subplot(2,2,1);
plot(m_final(end,1:end),P_final(end,1:end)/1e12,'-ob')
grid on;
subplot(2,2,2);
plot(m_final(end,1:end),rho_final(end,1:end),'-ob')
grid on;
subplot(2,2,3);
plot(m_final(end,1:end),As(end,1:end),'-ob')
% ylim([-1 1]*1e-2)
grid on;
subplot(2,2,4);
plot(m_final(end,1:end),At(end,1:end),'-ob')
ylim([-1 1]*1e0)
hold on;
grid on;


figure(2);
subplot(2,2,1);
plot(m_final(end,1:end),rho_final(end,1:end)/1e12,'-ob')
grid on;
subplot(2,2,2);
plot(m_final(end,1:end),size1(end,1:end),'-ob')
ylim([-2 1]*1e14);
grid on;
subplot(2,2,3);
plot(m_final(end,1:end),size2(end,1:end),'-ob')
ylim([-2 1]*1e13);
grid on;
subplot(2,2,4);
plot(m_final(end,1:end),size3(end,1:end),'-ob')
ylim([-2 1]*1e14);
hold on;
grid on;
