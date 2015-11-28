% simulate Emmanuel et al.'s example
clear all; close all; clc
% generate data 
X = [2 1 1 ; 1 1 2];
beta.truth = [0 1 0]';
y = X * beta.truth; 

% fit lasso 
[~,~, beta.lasso] = lasso_lsta(X,y, 1, 0.01, .1);
lasso