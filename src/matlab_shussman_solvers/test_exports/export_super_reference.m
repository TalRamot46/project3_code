% export_super_reference.m (script)
% cwd = test_exports. Uses super/my_Au.m and profiles_for_report_super-style formulas.
cd(fullfile('..', 'super'));
my_Au;
tau = 0;
[m0, mw, e0, ew, xsi, z, A, t, x] = manager(mat, tau);
times = 0.5;
T0 = 1;
m_heat = m0 * T0^mw(2) * times^mw(3) .* t' / xsi;
x_heat = m_heat / mat.rho0;
T_heat = T0 * times^tau * (x(:, 1));
xi = t(:) / xsi;
cd(fullfile('..', 'test_exports'));
save(fullfile(pwd, 'super_reference.mat'), 'm_heat', 'x_heat', 'T_heat', 'times', 'tau', 'T0', 'xi', 'xsi');
