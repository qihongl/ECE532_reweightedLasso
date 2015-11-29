% initialize 
clear all;close all;clc
% the data for the 1st subject, which includes the X matrix
load('data/jlp01.mat')           
% there are 10 structs in this metadata, the 1st struct corresponds to the
% 1st subject, which contains the y.
load('data/jlp_metadata.mat')    
[m,n] = size(X);    

% here's how you would retrieve the y vector
y = metadata(1).TrueFaces;  
y(y == 0) = -1;

%% normalize the features
% m = numPics, n = numVoxels
Xn = columnNormalization(X);

%% 
% cv
testSize = 50;
testIdx = false(m,1);
testIdx(1:testSize) = true;

% subset 
Xtrain = X(~testIdx,:);
ytrain = y(~testIdx);
Xtest = X(testIdx,:);
ytest = y(testIdx);

Xntrain = Xn(~testIdx,:);
Xntest = Xn(testIdx,:);

% set params
lambda = 1; 
tau = .9 / norm(X,2)^2;
tau_n = .9 / norm(Xn,2)^2;

%% fit the model 
[beta.raw, ~] = lasso_q(Xtrain, ytrain, lambda, tau, 0, 1);
[beta.normal, ~] = lasso_q(Xntrain, ytrain, lambda, tau_n, 0, 1);

% generate prediction  
predict = sign(Xtest * beta.raw(:,end)); 
predict_n = sign(Xntest * beta.normal(:,end)); 

accuracy = sum(bsxfun(@eq, predict, ytest))/testSize
accuracy_n = sum(bsxfun(@eq, predict_n, ytest))/testSize

%% print some results
plot(beta.raw(:,end), beta.normal(:,end), 'o')
fprintf('Difference in norm %f \n', norm(beta.raw(:,end) - beta.normal(:,end),1));
fprintf('Number of zero weights (raw vs. nor): %d %d\n', sum(beta.raw(:,end)==0), sum(beta.normal(:,end)==0));