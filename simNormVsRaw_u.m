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
% parameters for normalize data
tau_n = 1 / norm(X_n,2)^2;
% parameters for the raw data
tau = 1 / norm(X,2)^2;
% TODO: maybe we should tune the lambda for both methods here
lambda = 1;

% fit lasso to raw data
[~, ~, beta.raw] = lasso_lsta(X, y, lambda, tau, 0);
% fit lasso to normalized data
[~, ~, beta.normal] = lasso_lsta(X_n, y, lambda, tau_n, 0);

%% reweight by variance structure
diag(cov(X));

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

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% Helper functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% generate a beta vector
% input:    total number of features
%           number of informative features
%           the magnitude of the signal
% output:   a beta vector that statisfy you choices
function beta = generateBeta(numInfoFeatures, numTotalFeatures)
% generate the subset indices
temp = randperm(numTotalFeatures);
index = temp(1 : numInfoFeatures);
% set most of the beta to zero
beta = zeros(numTotalFeatures,1);
% set a subset of them to be randn
beta(index) = randn(size(index));
end