clear variables; close all; clc
% stimuli by voxel
m = 256;        % num stimuli
n = 520;        % num voxels

% generate X
X = randn(m,n);
% generate beta
beta.truth = generateBeta(130, n);
noise = 1 .* randn(m,1);
y = X * beta.truth;

[m,n] = size(X);
lambda = 1;

weights = ones(n,1);
for i = 1 : 10
% fit lasso
beta.rw = lasso_ista(X, y, lambda, weights, 0);
% update weights
errors = .1;
weights = 1 ./ (abs(beta.rw) + errors);

% record beta history 
history.weights(:,i) = weights;
history.beta(:,i) = beta.rw; 
diff = norm(beta.rw - beta.truth, inf);
fprintf('%d\t%f\n',i,diff);
end

plot(beta.truth, history.beta(:,10), 'o')

