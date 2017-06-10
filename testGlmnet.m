clear all;clc;
load('data/BreastCancer.mat')
% X = X(:,1:2000);
[M,N] = size(X);
testSize = 100; 
y(y == -1) = 0; 
% y = logical(y);

testIdx = false(M,1);
testIdx(randsample(M,testSize)) = true;

X_train = X(~testIdx,:);
X_test = X(testIdx,:);
y_train = y(~testIdx);
y_test = y(testIdx);

%%
clear options
options.alpha = 1;
options.nlambda = 100;

% fit = glmnet(X_train, y_train, 'binomial', options);
% sum(myStepFunction(X_test * fit.beta + fit.a0) == y_test) / length(y_test)
% numNonZeros(glmnetCoef(fit))
% glmnetPrint(fit)
% glmnetPlot(fit)

%%
cvfit = cvglmnet(X_train, y_train, 'binomial', options);
cvglmnetPlot(cvfit)
beta = cvglmnetCoef(cvfit, 'lambda_1se');
pred = cvglmnetPredict(cvfit, X_test);

sum(myStepFunction(pred == y_test) / length(y_test))



% y_train = logical(y_train);
% y_test = logical(y_test);
[B,FitInfo] = lassoglm(X_train,y_train,'binomial','NumLambda',10,'CV',10);

beta2 = B(:,FitInfo.IndexMinDeviance);
intercept = FitInfo.Intercept(FitInfo.IndexMinDeviance);
coef2 = [intercept; beta2];
preds = glmval(coef2,X_test,'logit');

sum(round(preds) == y_test) / length(y_test)
sum(myStepFunction(cvglmnetPredict(cvfit, X_test)) == y_test) / length(y_test);
