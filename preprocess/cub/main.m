fprintf('SfM for train\n');
cub_sfm('train');
fprintf('SfM for testval\n');
cub_sfm('testval');
fprintf('Splitting\n');
split_cub();
