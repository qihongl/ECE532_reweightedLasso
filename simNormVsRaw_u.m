function [out, finalBeta] = simNormVsRaw_u(signal, noise)
% stimuli by voxel 
m = 100;        % num stimuli
n = 200;        % num voxels
X = zeros(m,n); % preallocate data matrix 

% parameter
noise = 10;
signal = 10;

% generate X
X = randn(m,n);

% generate beta
beta_truth = generateBeta(20, n, 0);
noise = randn(m,1);
y = X * beta_truth + noise; 


%% fit the model 
% normalize data 
X_n = columnNormalization(X);
tau_n = 1 / norm(X_n,2)^2;

tau = 1 / norm(X,2)^2;

% TODO: maybe we should tune the lambda for both methods here
lambda = 1; 

% fit lasso to raw data
[~, ~, beta] = lasso_lsta(X, y, lambda, tau, 0);

% fit lasso to normalized data
[~, ~, beta_n] = lasso_lsta(X_n, y, lambda, tau_n, 0);

%% reweight by variance structure
diag(cov(X));

%% compute the different in beta estimates
% save the beta 
finalBeta.raw = beta;
finalBeta.norm = beta_n;

% save the different of the two betas
out.diff = norm(beta - beta_n);
out.spars.raw = nnz(beta);
out.spars.norm = nnz(beta_n);
out.trueBeta = beta_truth;
end

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%% Helper functions %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% generate a beta vector
% input:    total number of features 
%           number of informative features 
%           the magnitude of the signal             
% output:   a beta vector that statisfy you choices
function beta = generateBeta(numInfoFeatures, numTotalFeatures, signalSize)
sig = signalSize; %TODO
% generate the subset indices
temp = randperm(numTotalFeatures);
index = temp(1 : numInfoFeatures);
% set most of the beta to zero
beta = zeros(numTotalFeatures,1);
% set a subset of them to be randn 
beta(index) = randn(size(index));
end