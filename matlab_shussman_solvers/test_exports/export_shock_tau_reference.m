% export_shock_tau_reference.m (script)
% cwd must be test_exports. Shock with fixed P0 (Barye) and tau (Python patching_method=false).
cd(fullfile('..', 'shock'));
Au;
P0 = 2.71e12;
tau_drive = -0.45;
times = 1.0;
[m0, mw, e0, ew, u0, uw, xsi, z, utilda, ufront, t, x] = manager(mat, tau_drive);
t = flipud(t);
x = flipud(x);
tc = t(:);
m_shock = m0 * P0^mw(1) * times^mw(3) .* tc / xsi;
P_shock = P0 * times^tau_drive .* x(:, 2);
u_shock = u0 * P0^uw(1) * times^uw(3) .* x(:, 3) / utilda;
rho_shock = 1 ./ (mat.V0 * x(:, 1));
T_shock = (P_shock ./ mat.r ./ mat.f .* rho_shock.^(mat.mu - 1)).^(1 / mat.beta) / 1160500;
xi = tc / xsi;
cd(fullfile('..', 'test_exports'));
save(fullfile(pwd, 'shock_tau_reference.mat'), ...
    'm_shock', 'P_shock', 'u_shock', 'rho_shock', 'T_shock', 'xi', 'xsi', 'P0', 'tau_drive', 'times');
