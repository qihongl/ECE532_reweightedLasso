clear variables; close all; clc
% stimuli by voxel
m = 256;        % num stimuli
n = 520;        % num voxels

% generate X
X = randn(m,n);
% generate beta and y 
beta.truth = generateBeta(130, n);
y = X * beta.truth;

%% iteratively fitting reweighted-lasso
lambda = 1;
[beta.rw, history] = reweightedLasso(X, y, lambda, 1);

% display the final signal reconstruction alignment
plot(beta.truth, history.beta(:,end), 'o')

