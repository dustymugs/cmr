function n = quatnormalize(q)
%QUATNORMALIZE Normalize a quaternion.
%   N = QUATNORMALIZE(Q) computes N, the result of normalizing the
%   quaternion Q.
%
% from https://github.com/dhr/matlab-tools/tree/master/quaternions

n = bsxfun(@rdivide, q, sqrt(sum(q.^2, 2)));
