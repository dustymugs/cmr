function mean_shape(project_dir, split_name, kp_names, kp_perm, lr_edges, bf_edges, tb_edges)

    fprintf('Computing mean shape: %s\n', split_name)

    mean_shape_dir = fullfile(project_dir, 'sfm');
    mean_shape_path = fullfile(mean_shape_dir, ['anno_' split_name '.mat']);
    mkdirOptional(mean_shape_dir);

    data_file = fullfile(project_dir, 'data', [split_name '_cleaned.mat']);
    var = load(data_file);

    fprintf('Input File: %s\n', data_file)
    fprintf('Output File: %s\n', mean_shape_path)

    if ~exist(mean_shape_path)
        fprintf('Computing new mean shape\n')
        kps_all = [];
        vis_all = [];
        box_scale = [];

        %% Construct keypoint matrix
        fprintf('Construct keypoint matrix\n')
        n_objects = length(var.images);
        box_trans = zeros(n_objects, 2);
        fprintf('Processing objects')
        for b = 1:n_objects
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
            % keyboard;
            box_trans(b, :) = mean(kps_b(:, vis_b(1,:)>0), 2);
            box_trans_flipped = mean(kps_b_flipped(:, vis_b_flipped(1,:)>0), 2);

            kps_b = kps_b - box_trans(b, :)';
            kps_b_flipped = kps_b_flipped - box_trans_flipped;

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
        %S = diag([1 1 -1])*S; % preemptively flip
        while(~good_model)
            R = alignSfmModel(S, lr_edges, bf_edges, tb_edges);
            Srot = R*S;
            show3dModel(Srot, kp_names, 'convex_hull');
            user_in = input('Is this model aligned ? "y" will save and "n" will flip along the Z-axis: ','s');
            if(strcmp(user_in,'y'))
                good_model = 1;
                disp('Ok !')
            else
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
        fprintf('SfM of each object as transformation from mean shape\n')
        sfm_anno = struct;
        for bx = 1:n_objects
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
        fprintf('Compute convex hull\n')
        x = S(1, :)
        y = S(2, :)
        z = S(3, :)
        X = [x(:), y(:), z(:)]

        conv_tri_ = delaunayn(X, {'Qt', 'Qbb', 'Qc'});
        conv_tri = [conv_tri_(:, [1,2,3]); conv_tri_(:, [1,2,4]); conv_tri_(:, [1,3,4]); conv_tri_(:, [4,2,3])];
        fprintf('conv_tri:\n')
        disp(conv_tri)

        %save(mean_shape_path, 'sfm_anno', 'S', 'conv_tri', 'conv_tri_', 'X', '-v7');
        save(mean_shape_path, 'sfm_anno', 'S', 'conv_tri', '-v7');
    else
        fprintf('Loading existing sfm\n')
        load(mean_shape_path, 'sfm_anno', 'S',  'conv_tri');
    end
end


function [im, part] = load_image(root_dir, data)
    impath = fullfile(root_dir, 'images', data.rel_path);
    if exist(impath)
        im = myimread(impath);
    else
        img = ones(data.height, data.width, 3);
    end

    part = data.parts;
end

function im = myimread(impath)
    im = imread(impath);
    if size(im, 3) == 1
        im = repmat(im, [1,1,3]);
    end
end
