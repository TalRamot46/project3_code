% close all;
N=10000;
m_final=zeros(length(times),N);
P_final=zeros(length(times),N);
u_final=zeros(length(times),N);
T_final=zeros(length(times),N);
rho_final=1/mat.V0*ones(length(times),N);

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
end

hold on;
grid on;
plot(m_final(1,:),P_final(1,:), 'r-');
plot(m_heat(1,:),P_heat(1,:), 'g-');
plot(m_shock(1,:),P_shock(1,:), 'b-');
% plot(m_heat(1,:), rho_heat(1,:) / 30, 'c-')
% plot(m_shock(1,:), rho_shock(1,:) / 30, 'm-')
% plot(m_final(1,:),rho_final(1,:), 'r-');
% plot(m_heat(1,:),rho_heat(1,:), 'g-');
% plot(m_shock(1,:),rho_shock(1,:), 'b-');


% subplot(2,2,1);
% plot(m_final(end,1:end),P_final(end,1:end)/1e12,'-ob')
% grid on;
% subplot(2,2,2);
% plot(m_final(end,1:end),rho_final(end,1:end),'-ob')
% grid on;
% subplot(2,2,3);
% plot(m_final(end,1:end),As(end,1:end),'-ob')
% % ylim([-1 1]*1e-2)
% grid on;
% subplot(2,2,4);
% plot(m_final(end,1:end),At(end,1:end),'-ob')
% ylim([-1 1]*1e0)
% hold on;
% grid on;
% 
% 
% figure(2);
% subplot(2,2,1);
% plot(m_final(end,1:end),rho_final(end,1:end)/1e12,'-ob')
% grid on;
% subplot(2,2,2);
% plot(m_final(end,1:end),size1(end,1:end),'-ob')
% ylim([-2 1]*1e14);
% grid on;
% subplot(2,2,3);
% plot(m_final(end,1:end),size2(end,1:end),'-ob')
% ylim([-2 1]*1e13);
% grid on;
% subplot(2,2,4);
% plot(m_final(end,1:end),size3(end,1:end),'-ob')
% ylim([-2 1]*1e14);
% hold on;
% grid on;

