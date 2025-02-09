function [out] = simNormVsRaw_u(propUsefulVox, noiseLevel)
% stimuli by voxel
m = 100;        % num stimuli
n = 500;        % num voxels
nnzBeta = round(n * propUsefulVox);

% generate X
X = randn(m,n);
% generate beta
beta.truth = generateBeta(nnzBeta, n);
noise = noiseLevel .* randn(m,1);
y = X * beta.truth + noise;

% normalize the data by column 
X_n = columnNormalization(X);

%% fit the model
% TODO: maybe we should tune the lambda for both methods here
lambda = 1;
weights = ones(n,1);
% fit lasso to raw data
[beta.raw] = lasso_ista(X, y, lambda, weights, 0);
% fit lasso to normalized data
[beta.normal] = lasso_ista(X_n, y, lambda, weights, 0);

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

end