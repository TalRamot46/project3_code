% export_sub_reference.m (script)
% Run with: cd(fileparts(which('export_sub_reference.m')))  (i.e. cwd = test_exports)
% profiles_for_report_sub.m calls clear all — do not rely on script-local path variables afterward.
run(fullfile('..', 'sub', 'profiles_for_report_sub.m'));
save(fullfile(pwd, 'sub_reference.mat'), 'm_heat', 'P_heat', 'T_heat', 'u_heat', 'rho_heat', 'times', 't', 'xsi');
