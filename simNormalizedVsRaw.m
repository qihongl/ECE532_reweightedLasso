function [out, finalBeta] = simNormalizedVsRaw(signal, noise)
% stimuli by voxel 
m = 100;        % num stimuli
n = 200;        % num voxels
X = zeros(m,n); % preallocate data matrix 
y = [ones(m/2,1); -ones(m/2,1)];    % set true labels

% parameter
section = 50; 
% noise.big = 10;
% noise.small = 10;
% signal.big = 5;
% signal.small = 5;


%% generate data

% high noise, and magnitude 
X(y == 1, 1:section) = signal.big;
X(y ==-1, 1:section) = -signal.big;
X(:, 1:section) = X(:, 1:section) + randn(m, section) * noise.big;

% less noise, less magnitude 
X(y == 1, section+1:section*2) = signal.small;
X(y ==-1, section+1:section*2) = -signal.small;
X(:, section+1:section*2) = X(:, section+1:section*2) + randn(m, section) * noise.small;

% other noise
X(:,section*2+1:end) = randn(m, section*2) * noise.small;


%% fit the model 
% normalize data 
X_n = (X - mean(X(:))) / std(X(:));
tau_n = 1 / norm(X_n,2)^2;

tau = 1 / norm(X,2)^2;

% maybe we should tune the lambda for both methods here
lambda = 10; 

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

% record number of nonzeros betas in each section
out.raw.sec1 = nnz(beta(1:section));
out.raw.sec2 = nnz(beta(section+1: section*2));
out.raw.sec3 = nnz(beta(1+section*2:end));

out.norm.sec1 = nnz(beta_n(1:section));
out.norm.sec2 = nnz(beta_n(section+1: section*2));
out.norm.sec3 = nnz(beta_n(1+section*2:end));
end