clear variables; close all; 
load('data/BreastCancer.mat')

[m,n] = size(X);
Xn = columnNormalization(X);
lambda = 10; 
weights = ones(n,1);

%% fit regular lasso vs. reweighed lasso
[beta.std] = lasso_ista(X, y, lambda, weights, 0);
[beta.rw] = reweightedLasso(X, y, lambda, 1);

%% fit regular lasso vs. reweighed lasso (for normalized data)
[beta.normal] = lasso_ista(Xn, y, lambda, weights, 0);
[beta.normalrw] = reweightedLasso(Xn, y, lambda, 1);

%% find out performance
accuracy.raw = sum(sign(X * beta.std) == y )/ length(y)
accuracy.rawrw = sum(sign(X * beta.rw) == y )/ length(y)

accuracy.normal = sum(sign(X * beta.normal) == y) / length(y)
accuracy.normalrw = sum(sign(X * beta.normalrw) == y) / length(y)

nnz(beta.raw)
nnz(beta.rawrw)

nnz(beta.normal)
nnz(beta.normalrw)

% without normalization 
% accuracy  1   vs .94
% sparsity 3666 vs 142

%% plot
subplot(2,1,1)
plot(beta.raw)
title('Beta estimation with lasso WITHOUT reweighting')
subplot(2,1,2)
plot(beta.rawrw)
title('Beta estimation with lasso WITH reweighting')

subplot(2,1,1)
plot(beta.normal)
title('Beta estimation with lasso WITHOUT reweighting')
subplot(2,1,2)
plot(beta.normalrw)
title('Beta estimation with lasso WITH reweighting')