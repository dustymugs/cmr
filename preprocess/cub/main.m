% change this to your project directory
project_dir = '/cmr/cachedir/cub';

% change keypoints to your project
kp_names = {'Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye', 'LLeg', 'LWing', 'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat'};
kp_perm = [1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15];
kp_left_right = [8 12; 9 13]; % left to right edges (along -X) by index (not kp_perm values)
kp_back_front = [14 5]; % back to front edges (along -Y) (not kp_perm values)
kp_top_bottom = []; % top to bottom (along -Z)

% Back vs RLeg/LLeg


% always call init_cmr first
addpath('/cmr/preprocess/shape/cmr');
init_cmr()

mean_shape(project_dir, 'train', kp_names, kp_perm, kp_left_right, kp_back_front, kp_top_bottom);
mean_shape(project_dir, 'testval', kp_names, kp_perm, kp_left_right, kp_back_front, kp_top_bottom);
split_valtest(project_dir);
