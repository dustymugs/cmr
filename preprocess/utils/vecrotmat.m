function R = vecrotmat(v1, v2)
    % equivalent of MATLAB's vrrotvec2mat(vrrotvec(...))
    % https://octave.1599824.n4.nabble.com/find-the-rotation-matrix-between-two-vectors-tp2067390p2067807.html


    if (exist('vrrotvec') ~= 0)
        R = vrrotvec2mat(vrrotvec(v1, v2));
    else
        % Get the axis and angle
        angle = acos(v1' * v2);
        axis = cross(v1, v2) / norm(cross(v1, v2));

        % A skew symmetric representation of the normalized axis
        axis_skewed = [0 -axis(3) axis(2) ; axis(3) 0 -axis(1) ; -axis(2) axis(1) 0];

        % Rodrigues formula for the rotation matrix
        R = eye(3) + sin(angle) * axis_skewed + (1 - cos(angle)) * axis_skewed * axis_skewed;
    end
end
