addpath('../cmr');
addpath('../utils');
addpath('../sfm');
addpath('../quaternions');

if (exist('OCTAVE_VERSION', 'builtin') ~= 0)
    graphics_toolkit('gnuplot')
    setenv('GNUTERM', 'x11')

    %debug_on_warning(1);
    debug_on_error(1);

    page_output_immediately(1);
    page_screen_output(0);

    addpath('../octave');
end

fprintf('SfM for train\n')
mean_shape('train');
fprintf('SfM for testval\n')
mean_shape('testval');
fprintf('Splitting\n')
split_valtest();
