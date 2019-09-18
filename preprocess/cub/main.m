addpath('/cmr/preprocess/shape/cmr');
addpath('/cmr/preprocess/shape/utils');
addpath('/cmr/preprocess/shape/sfm');
addpath('/cmr/preprocess/shape/quaternions');

if (exist('OCTAVE_VERSION', 'builtin') ~= 0)
    graphics_toolkit('gnuplot')
    setenv('GNUTERM', 'x11')

    %debug_on_warning(1);
    debug_on_error(1);

    page_output_immediately(1);
    page_screen_output(0);

    addpath('/cmr/preprocess/shape/octave');
end

% change keypoints to your project
kp_names = {'Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye', 'LLeg', 'LWing', 'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat'};
kp_perm = [1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15];

% change this to your project directory
project_dir = '/cmr/cachedir/cub';

mean_shape(project_dir, 'train', kp_names, kp_perm);
mean_shape(project_dir, 'testval', kp_names, kp_perm);
split_valtest(project_dir);
