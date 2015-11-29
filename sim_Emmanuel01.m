% replicate Emmanuel et al.'s example
clear variables; close all; clc
% generate data 
X = [2 1 1 ; 1 1 2];
beta.truth = [0 1 0]';
y = X * beta.truth; 

[m,n] = size(X);
lambda = 1; 

% plain lasso
weights = [1 1 1]'; % no re-weighting
beta.raw = lasso_ista(X, y, lambda, weights, 0);

% reweighted lasso
weights = [3 1 3]';
beta.rw = lasso_ista(X, y, lambda, weights, 0);

% compare both estimated parameters to the truth
[beta.raw beta.rw beta.truth]



