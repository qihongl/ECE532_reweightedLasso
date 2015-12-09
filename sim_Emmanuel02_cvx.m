function [dev, nnzs] = sim_Emmanuel02_cvx()
% clear variables; close all; clc
% stimuli by voxel
m = 256;        % num stimuli
n = 512;        % num voxels

% generate X
X = randn(m,n);
% generate beta and y
beta.truth = generateBeta(130, n, 1);
noise = randn(m,1) * .1;
y = X * beta.truth + noise;

%% fitting reweighted-lasso 
[beta.rw history] = reweightedLasso_cvx(X,y, 1e-6);
% fit regular lasso
beta.lasso = lasso_ista(X,y,1,ones(n,1),0);

beta.ls = pinv(X) * y; 

%%
nnzs.rw = numNonZeros(beta.rw);
nnzs.lasso = numNonZeros(beta.lasso);
nnzs.ls = numNonZeros(beta.ls);

dev.beta.rw = norm(beta.truth - beta.rw);
dev.beta.lasso = norm(beta.truth - beta.lasso);
dev.beta.ls = norm(beta.truth - beta.ls);

dev.y.rw1 = norm(y - X * history.beta(:,1));
dev.y.rw = norm(y - X * beta.rw);
dev.y.lasso = norm(y - X * beta.lasso); 
dev.y.ls = norm(y - X * beta.ls);

%% paper plot 
plotPerformance(beta.truth, history, X,y)
end