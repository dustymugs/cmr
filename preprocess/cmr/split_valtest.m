function split_valtest(project_name)
    % Splits the test set into val / test.

    fprintf('Splitting valtest: %s\n', project_name)

    cache_dir = fullfile(pwd, '..', '..', 'cachedir', project_name);

    orig_path = fullfile(cache_dir, 'data', 'testval_cleaned.mat');
    orig_sfm_path = fullfile(cache_dir, 'sfm', 'anno_testval.mat');

    % New mat.
    val_path = fullfile(cache_dir, 'data', 'val_cleaned.mat');
    test_path = fullfile(cache_dir, 'data', 'test_cleaned.mat');
    val_sfm_path = fullfile(cache_dir, 'sfm', 'anno_val.mat');
    test_sfm_path = fullfile(cache_dir, 'sfm', 'anno_test.mat');

    % Load all data. This is already cleaned
    load(orig_path, 'images');
    load(orig_sfm_path, 'sfm_anno', 'S');

    num_images = length(images);

    half = round(num_images / 2);

    rng(100);

    inds = randperm(num_images);
    test_inds = sort(inds(1:half));
    val_inds = sort(inds(half:end));

    test_images = images(test_inds);
    val_images = images(val_inds);

    test_sfm_anno = sfm_anno(test_inds);
    val_sfm_anno = sfm_anno(val_inds);

    save_mat(test_path, test_images);
    save_mat(val_path, val_images);

    save_sfm_mat(test_sfm_path, test_sfm_anno);
    :ave_sfm_mat(val_sfm_path, val_sfm_anno);
end

function save_mat(mat_path, images)
    save(mat_path, 'images');
end

function save_sfm_mat(mat_path, sfm_anno)
    save(mat_path, 'sfm_anno');
end
