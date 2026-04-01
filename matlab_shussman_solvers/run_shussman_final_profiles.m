function run_shussman_final_profiles(output_path)
%RUN_SHUSSMAN_FINAL_PROFILES Run subsonic then shock report scripts and save the pressure figure.
%
%   run_shussman_final_profiles()
%   run_shussman_final_profiles('C:\path\to\out.png')
%
% Order matches the manual workflow:
%   1. sub/profiles_for_report_sub.m  (must run from sub/ so sub Au + manager resolve)
%   2. shock/profiles_for_report_shock.m (must run from shock/)
%   3. shock/my_final_profiles.m (figure: combined vs heat vs shock pressure vs m)
%
% Requires MATLAB with -batch-capable release (R2019a+) for non-interactive use.

root = fileparts(mfilename('fullpath'));
if nargin < 1 || isempty(output_path)
    output_path = fullfile(root, 'shussman_pressure_profiles.png');
end

subdir = fullfile(root, 'sub');
shockdir = fullfile(root, 'shock');

if ~isfolder(subdir) || ~isfolder(shockdir)
    error('run_shussman_final_profiles:MissingFolder', ...
        'Expected sub/ and shock/ under %s', root);
end

% profiles_for_report_sub.m starts with clear all; it clears this function's workspace.
% Keep the output path in the base workspace so it survives; recompute folders via mfilename.
assignin('base', 'runShussmanOutputPath', output_path);

% run(fullfile(...)) puts each script's folder on the path for Au.m / manager.m
fprintf("running sub...\n")
run(fullfile(subdir, 'profiles_for_report_sub.m'));

root = fileparts(mfilename('fullpath'));
shockdir = fullfile(root, 'shock');
fprintf("running shock...\n")
run(fullfile(shockdir, 'profiles_for_report_shock.m'));
fprintf("running final...\n")
run(fullfile(shockdir, 'my_final_profiles.m'));

fprintf("presenting output...\n")
output_path = evalin('base', 'runShussmanOutputPath');
evalin('base', 'clear runShussmanOutputPath');

f = gcf;
if verLessThan('matlab', '9.8') % R2020a
    print(f, output_path, '-dpng', '-r150');
else
    exportgraphics(f, output_path, 'Resolution', 150);
end

fprintf('Saved figure to: %s\n', output_path);
end
