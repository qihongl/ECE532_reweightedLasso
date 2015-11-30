% initialize 
clear all;close all;clc
% the data for the 1st subject, which includes the X matrix
load('data/jlp01.mat')           
% there are 10 structs in this metadata, the 1st struct corresponds to the
% 1st subject, which contains the y.
load('data/jlp_metadata.mat')    
[m,n] = size(X);    

% simulation parameters
noiseLevel = .2;  
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
% TODO: maybe we should tune the lambda for both methods here
lambda = 1;
weights = ones(n,1);
% fit lasso to raw data
[beta.raw] = lasso_ista(X, y, lambda, weights, false);
[beta.rawrw] = reweightedLasso(X, y, lambda, true);

% fit lasso to normalized data
[beta.normal] = lasso_ista(X_n, y, lambda, weights, false);
[beta.normalrw] = reweightedLasso(X_n, y, lambda, true);

% %% compute some useful indicators
% out.beta = beta;
% % number of non-zero parameters
% out.spars.raw = nnz(beta.raw);
% out.spars.norm = nnz(beta.normal);
% % y hat deviation from true y
% out.yDevation.raw = norm(y - X * beta.raw,2);
% out.yDevation.norm = norm(y - X_n * beta.normal,2);
% % beta hat deviation from true beta
% out.betaDevation.raw = norm(beta.raw - beta.truth,2);
% out.betaDevation.norm = norm(beta.normal - beta.truth,2);



%% comparison
FS = 15;
% compare differences for weighted lasso and unweighted lasso
subplot(2,3,1)
plot(beta.raw,beta.normal, '.')
title('raw vs. normalized: without reweighting', 'fontsize', FS)
ylabel('estimates, raw data', 'fontsize', FS)
xlabel('estimates, normalized ', 'fontsize', FS)

subplot(2,3,4)
plot(beta.rawrw,beta.normalrw, '.')
title('raw vs. normalized: with reweighting', 'fontsize', FS)
ylabel('estimates, raw data', 'fontsize', FS)
xlabel('estimates, normalized ', 'fontsize', FS)


% compare weighted estimation to unweighted
subplot(2,3,2)
plot(beta.rawrw, beta.raw, '.')
title('with vs. without reweighting: raw data', 'fontsize', FS)
ylabel('estimates without reweighting', 'fontsize', FS)
xlabel('estimates with reweighting', 'fontsize', FS)

subplot(2,3,5)
plot(beta.normalrw, beta.normal, '.')
title('with vs. without reweighting: normalized data', 'fontsize', FS)
ylabel('estimates without reweighting', 'fontsize', FS)
xlabel('estimates with reweighting', 'fontsize', FS)

% compare estimation to truth 
subplot(2,3,3)
plot(beta.truth,beta.rawrw, '.')
title('truth vs. reweighting, raw data', 'fontsize', FS)
ylabel('Reweighted estimates', 'fontsize', FS)
xlabel('true beta', 'fontsize', FS)

subplot(2,3,6)
plot(beta.truth,beta.normalrw, '.')
ylabel('Reweighted estimates', 'fontsize', FS)
title('truth vs. reweighting, normalized data', 'fontsize', FS)
xlabel('true beta', 'fontsize', FS)