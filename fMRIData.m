% load the fMRI data
clear variables;close all;clc
% the data for the 1st subject, which includes the X matrix
load('data/jlpData.mat') 
% X = X(:,1:3000);
[m,n] = size(X);

Xn = columnNormalization(X);

%% Set Cross validation set 
testSize = 50;
testIdx = false(m,1);
testIdx(randsample(m,testSize)) = 1;

% subset raw data  
Xtrain = X(~testIdx,:);
ytrain = y(~testIdx);
Xtest = X(testIdx,:);
ytest = y(testIdx);
% subset normalized data
Xntrain = Xn(~testIdx,:);
Xntest = Xn(testIdx,:);

%% fit the lasso model 
% set params
LAMDBAS = logspace(-1,6, 50);
for i = 1: numel(LAMDBAS)
    [beta.raw(:,i)] = lasso_ista(Xtrain, ytrain, LAMDBAS(i), ones(n,1), 0);
    [beta.norm(:,i)] = lasso_ista(Xntrain, ytrain, LAMDBAS(i), ones(n,1), 0);
end


%% fit the reweighted lasso model 
sampleSize = 15; 
DELTA = logspace(-1, 2, sampleSize);
history = cell(sampleSize,1);
for i = 1 : numel(DELTA)
    fprintf('Rw: %3d\n',i);
    [beta.rawrw(:,i), history{i}] = reweightedLasso_cvx(Xtrain, ytrain, DELTA(i));
    [beta.normrw(:,i), history{i}] = reweightedLasso_cvx(Xntrain, ytrain, DELTA(i));
end

%% plot the data 
for i = 1 : numel(LAMDBAS)
    predict.raw(:,i) = sign(Xtest * beta.raw(:,i)); 
    accuracy.raw(i) = sum(predict.raw(:,i) == ytest) / testSize;
    nnzs.lasso(i) = numNonZeros(beta.raw(:,i));    
end

plot(nnzs.lasso,accuracy.raw, 'linewidth', 1.5)
title('LASSO: Accuracy-Sparsity trade off', 'fontsize' ,14)
xlabel('Number of non-zero feautres', 'fontsize' ,14)
ylabel('Cross Validated Accuracy', 'fontsize' ,14)



%% plot the data 
% for i = 1 : numel(DELTA)
%     predict.rawrw(:,i) = sign(Xtest * beta.rawrw(:,i)); 
%     accuracy.rawrw(i) = sum(predict.rawrw(:,i) == ytest) / testSize;
%     nnzs.rawrw(i) = numNonZeros(beta.rawrw(:,i));    
% end
% 
% % plot 
% plot(nnzs.rawrw, accuracy.rawrw, 'linewidth', 1.5)
% title('Reweighted LASSO: Accuracy-Sparsity trade off', 'fontsize' ,14)
% xlabel('Number of non-zero feautres', 'fontsize' ,14)
% ylabel('Cross Validated Accuracy', 'fontsize' ,14)

% save x, y, history, beta, delta