clear variables; close all; clc
% stimuli by voxel
m = 256;        % num stimuli
n = 520;        % num voxels
rng(1)
% generate X
X = randn(m,n);

% generate beta and y
beta.truth = generateBeta(100, n);
% beta.truth(beta.truth~=0) = 1;

y = X * beta.truth;

%% iteratively fitting reweighted-lasso
LAMBDAS = logspace(-2,2,20);
history = cell(numel(LAMBDAS),1);

for i = 1 : numel(LAMBDAS)
    [beta.rw(:,i), history] = reweightedLasso(X, y, LAMBDAS(i), 0);
    nnzb(i) = nnz(beta.rw(:,i));
end


%% plot 


%% plot
% display the final signal reconstruction alignment
subplot(1,2,1)
plot(nnzb, 'linewidth', 2)
title('sparsity vs. lambda')
xlabel('Increasing lambda'); ylabel('Number of non-zero beta')

% compute the difference between estimate vs. truth over iterations
diff = bsxfun(@minus, beta.rw, beta.truth);

for i = 1 : numel(LAMBDAS)
    diffnorm(i) = norm(diff(:,i),2);
end

subplot(1,2,2)
plot(diffnorm, 'linewidth', 2)
title('signal reconstruction deviation vs. lambda')
xlabel('Increasing lambda')
ylabel('Norm(reweighted estimation of beta - truth beta)')
