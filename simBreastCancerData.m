clear all; close all; clc
load('data/BreastCancer.mat')



tau = .9 / norm(X,2)^2;
lambda = .1; 

[finalBeta.std, ~] = lasso_q(X, y, lambda, tau, false, 1);
[finalBeta.rw, ~] = lasso_q(X, y, lambda, tau, true, 1);




