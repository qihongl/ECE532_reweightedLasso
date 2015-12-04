%% initialization 
clear variables; close all;
% load('data/BreastCancer.mat')
% [m,n] = size(X);

m = 100; n = 200;
X = randn(m,n);
beta.truth = 2 * generateBeta(50,n);
y = sign(X * beta.truth);


wts = ones(n,1);
lambda = 1;

%% fit lasso with cvx
cvx_begin quiet
    cvx_precision low
    variable beta_hat(n)
    minimize(norm(X*beta_hat - y,2) + lambda*norm(beta_hat ,1))
cvx_end
beta.cvx = beta_hat;

% fit lasso with ista
beta.ista = lasso_ista(X,y, 1, wts, 0);

%% check 
struct2array(beta)

sum(y == sign(X * beta.ista)) / length(y)
sum(y == sign(X * beta.cvx)) / length(y)

numNonZeros(beta.ista)
numNonZeros(beta.cvx)

plot(beta.ista, beta.cvx, 'o')

% plot(beta.ista, beta.truth, 'o')
% plot(beta.cvx, beta.truth, 'o')