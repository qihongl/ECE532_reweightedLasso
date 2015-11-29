% initialize 
clear all;close all;clc
% the data for the 1st subject, which includes the X matrix
load('data/jlp01.mat')           
% there are 10 structs in this metadata, the 1st struct corresponds to the
% 1st subject, which contains the y.
load('data/jlp_metadata.mat')    
[m,n] = size(X);    

% simulation parameters
noiseLevel = 1;  
propUsefulVox = 0.1;
nnzBeta = round(n * propUsefulVox);

% generate beta and y 
beta.truth = generateBeta(nnzBeta, n);
noise = noiseLevel .* randn(m,1);
y = X * beta.truth;

%% normalize the features
% m = numPics, n = numVoxels
X_n = columnNormalization(X);


%% fit the model
% parameters for normalize data
tau_n = 1 / norm(X_n,2)^2;
% parameters for the raw data
tau = 1 / norm(X,2)^2;
% TODO: maybe we should tune the lambda for both methods here
lambda = 1;

% fit lasso to raw data
[beta.raw] = lasso_q(X, y, lambda, tau, false, 1);
% [beta.rawrw] = lasso_q(X, y, lambda, tau, true, 0);
% fit lasso to normalized data
[beta.normal] = lasso_q(X_n, y, lambda, tau_n, false ,1);
% [beta.normalrw] = lasso_q(X_n, y, lambda, tau_n, true, 0);

%% compute some useful indicators
out.beta = beta;
% number of non-zero parameters
out.spars.raw = nnz(beta.raw);
out.spars.norm = nnz(beta.normal);
% y hat deviation from true y
out.yDevation.raw = norm(y - X * beta.raw,2);
out.yDevation.norm = norm(y - X_n * beta.normal,2);
% beta hat deviation from true beta
out.betaDevation.raw = norm(beta.raw - beta.truth,2);
out.betaDevation.norm = norm(beta.normal - beta.truth,2);

