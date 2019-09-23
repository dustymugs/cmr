function split_dataset(project_dir, in_split, out1_split, out2_split, percent_out1, split_sfm)
    % Splits input into two output splits

    if nargin < 5
        percent_out1 = 0.5
    end

    if nargin < 6
        split_sfm = true
    end

    assert((percent_out1 > 0.) && (percent_out1 < 1.))

    fprintf(
        'Splitting from "%s" to "%s" (%g%%) and "%s" (%g%%)\n',
        in_split,
        out1_split,
        percent_out1 * 100,
        out2_split,
        (1 - percent_out1) * 100
    )

    in_path = fullfile(project_dir, 'data', [in_split '_cleaned.mat']);

    % New mat.
    out1_path = fullfile(project_dir, 'data', [out1_split '_cleaned.mat']);
    out2_path = fullfile(project_dir, 'data', [out2_split '_cleaned.mat']);

    % Load all data
    load(in_path, 'images');

    num_images = length(images);
    split_idx = round(num_images * percent_out1);

    rng(100);
    inds = randperm(num_images);

    out1_inds = sort(inds(1:split_idx));
    out2_inds = sort(inds(split_idx:end));

    out1_images = images(out1_inds);
    out2_images = images(out2_inds);

    save_mat(out1_path, out1_images);
    save_mat(out2_path, out2_images);

    if split_sfm
        in_sfm_path = fullfile(project_dir, 'sfm', ['anno_' in_split '.mat']);
        out1_sfm_path = fullfile(project_dir, 'sfm', ['anno_' out1_split '.mat']);
        out2_sfm_path = fullfile(project_dir, 'sfm', ['anno_' out2_split '.mat']);

        load(in_sfm_path, 'sfm_anno', 'S');
        out1_sfm_anno = sfm_anno(out1_inds);
        out2_sfm_anno = sfm_anno(out2_inds);
        save_sfm_mat(out1_sfm_path, out1_sfm_anno);
        save_sfm_mat(out2_sfm_path, out2_sfm_anno);
    end
end

function save_mat(mat_path, images)
    save(mat_path, 'images', '-v7');
end

function save_sfm_mat(mat_path, sfm_anno)
    save(mat_path, 'sfm_anno', '-v7');
end
