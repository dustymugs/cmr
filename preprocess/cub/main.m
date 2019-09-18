addpath('/cmr/preprocess/shape/cmr');

% change keypoints to your project
kp_names = {'Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye', 'LLeg', 'LWing', 'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat'};
kp_perm = [1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15];

% change this to your project directory
project_dir = '/cmr/cachedir/cub';

% always call init_cmr first
init_cmr()

mean_shape(project_dir, 'train', kp_names, kp_perm);
mean_shape(project_dir, 'testval', kp_names, kp_perm);
split_valtest(project_dir);
