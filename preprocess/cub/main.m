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

kp_names = {'Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye', 'LLeg', 'LWing', 'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat'};
kp_perm = [1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15];

mean_shape('cub', 'train', kp_names, kp_perm);
mean_shape('cub', 'testval', kp_names, kp_perm);
split_valtest('cub');
