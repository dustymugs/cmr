addpath('../utils');
addpath('../sfm');
addpath('../quaternions');

if (exist('OCTAVE_VERSION', 'builtin') ~= 0)
    debug_on_warning(1);
    debug_on_error(1);
    page_output_immediately(1);
    page_screen_output(0);
    addpath('../octave');
end

fprintf('SfM for train\n')
cub_sfm('train');
fprintf('SfM for testval\n')
cub_sfm('testval');
fprintf('Splitting\n')
split_cub();
