clear variables; close all; 
load('data/BreastCancer.mat')

numFeatures = 512; 
numEgs = 256;
[M,N] = size(X);
egidx = randsample(M,numEgs);
X = X(egidx, randsample(N,numFeatures));
y=y(egidx);
[m,n] = size(X);  


%% Cross validation
% indices for the testing set
testSize = 45;
testIdx = false(m,1);
testIdx(randsample(m,testSize)) = 1;

% split the data
Xtrain = X(~testIdx,:);
ytrain = y(~testIdx);
Xtest = X(testIdx,:);
ytest = y(testIdx);



%% fitting reweighted-lasso 
[beta.rw history] = reweightedLasso_cvx(Xtrain,ytrain);

beta.lasso = lasso_ista(Xtrain,ytrain,1,ones(n,1),0);


%% plot
% if the truth signal have similar magnitude
plot(beta.rw, beta.lasso, 'o')
title('Compare rw and regular lasso')

sum(sign(Xtest * beta.rw) == ytest) / testSize
sum(sign(Xtest * beta.lasso) == ytest)  / testSize

numNonZeros(beta.rw)
numNonZeros(beta.lasso)

% 
% L0 find a much sparser solution than lasso! 
% 

