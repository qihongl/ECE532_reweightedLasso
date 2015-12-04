%% initialization 
clear variables; close all;clc;

X = [2 1 1 ; 1 1 2];
beta.truth = [0 1 0]';
y = X * beta.truth; 

[m,n] = size(X);

% 
wts = ones(n,1);
lambda = 1;
wts = [3 1 3]';

%% fit lasso with cvx
cvx_begin 
    variable beta_hat(n)
    minimize(lambda*norm(wts.*beta_hat ,1))
    subject to 
        X * beta_hat == y
cvx_end
beta.cvx = beta_hat;


%% check 
struct2array(beta)

sum(y == sign(X * beta.cvx)) / length(y)

numNonZeros(beta.cvx)

plot(beta.cvx, beta.truth, 'o')