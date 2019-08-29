function cub_sfm(split_name)

cub_cache_dir = fullfile(pwd, '..', '..', 'cachedir', 'cub');

out_dir = fullfile(cub_cache_dir, 'sfm');
out_path = fullfile(out_dir, ['anno_' split_name '.mat']);
mkdirOptional(out_dir);

cub_file = fullfile(cub_cache_dir, 'data', [split_name '_cub_cleaned.mat']);

kp_names = {'Back', 'Beak', 'Belly', 'Breast', 'Crown', 'FHead', 'LEye', 'LLeg', 'LWing', 'Nape', 'REye', 'RLeg', 'RWing', 'Tail', 'Throat'};
kp_perm = [1, 2, 3, 4, 5, 6, 11, 12, 13, 10, 7, 8, 9, 14, 15];
var = load(cub_file);

fprintf('CUB File: %s\n', cub_file)
fprintf('Out Path: %s\n', out_path)

if ~exist(out_path)
    fprintf('Computing new sfm\n')
    kps_all = [];
    vis_all = [];
    box_scale = [];

    lr_edges = [8 12; 9 13]; %left to right edges (along -X)
    bf_edges = [14 5]; % back to front edges (along -Y)

    %% Construct keypoint matrix
    n_birds = length(var.images);
    box_trans = zeros(n_birds, 2);
    fprintf('Processing birds')
    for b = 1:n_birds
        % bbox to normalize
        bbox_h = var.images(b).bbox.y2 - var.images(b).bbox.y1 + 1;
        bbox_w = var.images(b).bbox.x2 - var.images(b).bbox.x1 + 1;
        box_scale(b) = max(bbox_w, bbox_h);
        kps_b = var.images(b).parts(1:2, :)/box_scale(b);
        % Add flipped data
        kps_b_flipped = kps_b(:, kp_perm);
        kps_b_flipped(1, :) = -kps_b_flipped(1, :);

        vis_b = vertcat(var.images(b).parts(3, :), var.images(b).parts(3, :));
        vis_b_flipped = vis_b(:, kp_perm);

        % Mean center here,,
        box_trans(b, :) = mean(kps_b(:, vis_b(1,:)>0), 2);
        kps_b = kps_b - box_trans(b, :)';
        kps_b_flipped = kps_b_flipped - mean(kps_b_flipped(:, vis_b_flipped(1,:)>0), 2);
        
        kps_all = vertcat(kps_all, kps_b, kps_b_flipped);

        vis_all = vertcat(vis_all, vis_b, vis_b_flipped);
        % keyboard
        % sfigure(2); clf;
        % scatter(kps_b(1,:), kps_b(2,:));
        % hold on;
        % scatter(kps_b_flipped(1,:), kps_b_flipped(2,:));
        if (mod(b, 100) == 0)
            fprintf('.')
        end
    end
    fprintf('.\n')

    %% Compute mean shape and poses
    fprintf('Compute mean shape and poses\n')
    kps_all(~vis_all) = nan;
    [~, S, ~] = sfmFactorization(kps_all, 30, 10);
    % show3dModel(S, kp_names, 'convex_hull');
    %cameratoolbar

    %% Align mean shape to canonical directions
    fprintf('Align mean shape to canonical directions\n')
    good_model = 0;
    flip = 0;
    while(~good_model)
        R = alignSfmModel(S, lr_edges, bf_edges, []);
        Srot = R*S;
        show3dModel(Srot, kp_names, 'convex_hull');
        user_in = input('Is this model aligned ? "y" will save and "n" will realign after flipping: ','s');
        if(strcmp(user_in,'y'))
            good_model = 1;
            disp('Ok !')
        else
            flip = mod(flip+1,2);
            S = diag([1 1 -1])*S;        
        end
        close all;
    end
    S = Srot;
    max_dist = max(pdist(S'));
    S_scale = 2. / max_dist;
    fprintf('Scale Shape by %.2g\n', S_scale)
    S = S*S_scale;
    [M,T,~] = sfmFactorizationKnownShape(kps_all, S, 50);

    %%
    fprintf('SfM Annotations\n')
    sfm_anno = struct;
    for bx = 1:n_birds
        b = 2*bx-1;
        motion = M([2*b-1, 2*b], :);
        scale = norm(motion(1,:));
        rot = motion/scale;
        rot = [rot;cross(rot(1,:),rot(2,:))];
        if(det(rot)<0)
            rot(3,:) = -rot(3,:);
        end
        % reproj = motion * S + T([2*b-1, 2*b], :);
        % reproj2 = rot * S;
        % reproj2 = scale * (reproj2(1:2, :)) + T([2*b-1, 2*b], :);
        % norm(reproj - reproj2);
        [scale, rot, trans] = reprojMinimize(kps_all([2*b-1, 2*b], :), S, scale, rot, T([2*b-1, 2*b], :));
        sfm_anno(bx).rot = rot;
        sfm_anno(bx).scale = scale*box_scale(bx);
        sfm_anno(bx).trans = trans'*box_scale(bx) + box_trans(bx,:)'*box_scale(bx);
    end
    
    %% Compute and save convex hull
    fprintf('Compute and save convex hull\n')
    x = S(1, :)
    y = S(2, :)
    z = S(3, :)
    X = [x(:), y(:), z(:)]

    conv_tri_ = delaunayn(X, {'Qt', 'Qbb', 'Qc'});
    conv_tri = [conv_tri_(:, [1,2,3]); conv_tri_(:, [1,2,4]); conv_tri_(:, [1,3,4]); conv_tri_(:, [4,2,3])];
    fprintf('conv_tri:\n')
    disp(conv_tri)

    save(out_path, 'sfm_anno', 'S', 'conv_tri', 'conv_tri_', 'X', '-v7');
else
    fprintf('Loading existing sfm\n')
    load(out_path, 'sfm_anno', 'S',  'conv_tri');
end


function [im, part] = load_image(cub_dir, data)
impath = fullfile(cub_dir, 'images', data.rel_path);
if exist(impath)
    im = myimread(impath);
else
    img = ones(data.height, data.width, 3);
end

part = data.parts;

function im =  myimread(impath)
im = imread(impath);
if size(im, 3) == 1
    im = repmat(im, [1,1,3]);
end
