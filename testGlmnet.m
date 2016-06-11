clear all;clc;
load('data/BreastCancer.mat')

[M,N] = size(X);
testSize = 100; 

testIdx = false(M,1);
testIdx(randsample(M,testSize)) = true;

X_train = X(~testIdx,:);
X_test = X(testIdx,:);
y_train = y(~testIdx);
y_test = y(testIdx);

%%
clear options
options.alpha = 1;
options.nlambda = 300;

% fit = glmnet(X_train, y_train, 'binomial', options);
% sum(myStepFunction(X_test * fit.beta + fit.a0) == y_test) / length(y_test)
% numNonZeros(glmnetCoef(fit))
% glmnetPrint(fit)
% glmnetPlot(fit)

%%
cvfit = cvglmnet(X_train, y_train, 'binomial', options);
cvglmnetPlot(cvfit)
beta = cvglmnetCoef(cvfit, 'lambda_1se');
sum(myStepFunction(cvglmnetPredict(cvfit, X_test)) == y_test) / length(y_test);

