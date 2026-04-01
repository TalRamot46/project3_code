% export_piecewise_m_P.m (script)
% cwd = test_exports. Chained sub -> shock -> my_final_profiles.
run(fullfile('..', 'sub', 'profiles_for_report_sub.m'));
run(fullfile('..', 'shock', 'profiles_for_report_shock.m'));
run(fullfile('..', 'shock', 'my_final_profiles.m'));
m_fp = m_final(1, :);
P_fp = P_final(1, :);
save(fullfile(pwd, 'piecewise_m_P_reference.mat'), 'm_fp', 'P_fp');
