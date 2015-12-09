clear variables; close all; clc
% stimuli by voxel
m = 256;        % num stimuli
n = 512;        % num voxels
X = randn(m,n);

% generate beta and y 
beta.truth = generateBeta(100, n, 1);
noise = randn(m,1);
y = X * beta.truth;

%% iteratively fitting reweighted-lasso
lambda = 1;
[beta.rw, history] = reweightedLasso(X, y, true);
[beta.lasso] = lasso_ista(X,y, lambda, ones(n,1), 0);


%% paper plot 
plotPerformance(beta.truth, history,X,y)

