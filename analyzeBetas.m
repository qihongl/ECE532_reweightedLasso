%% plot the data 
for i = 1 : numel(LAMDBAS)
    predict.raw(:,i) = sign(Xtest * beta.raw(:,i)); 
    predict.norm(:,i) = sign(Xntest * beta.norm(:,i)); 
    accuracy.raw(i) = sum(predict.raw(:,i) == ytest) / testSize;
    accuracy.norm(i) = sum(predict.norm(:,i) == ytest) / testSize;
    nnzs.raw(i) = numNonZeros(beta.raw(:,i));    
    nnzs.norm(i) = numNonZeros(beta.norm(:,i));    
end

subplot(1,2,1)
plot(nnzs.raw, accuracy.raw, 'linewidth', 1.5)
title('LASSO: Accuracy-Sparsity trade off', 'fontsize' ,14)
xlabel('Number of non-zero feautres', 'fontsize' ,14)
ylabel('Cross Validated Accuracy', 'fontsize' ,14)
subplot(1,2,2)
plot(nnzs.norm, accuracy.norm, 'linewidth', 1.5)

%% plot the data 
for i = 1 : numel(DELTA)
    predict.rawrw(:,i) = sign(Xtest * beta.rawrw(:,i)); 
    predict.normrw(:,i) = sign(Xntest * beta.normrw(:,i)); 
    accuracy.rawrw(i) = sum(predict.rawrw(:,i) == ytest) / testSize;
    accuracy.normrw(i) = sum(predict.normrw(:,i) == ytest) / testSize;
    nnzs.rawrw(i) = numNonZeros(beta.rawrw(:,i));    
    nnzs.normrw(i) = numNonZeros(beta.normrw(:,i));    
end

% plot 
subplot(1,2,1)
plot(nnzs.rawrw, accuracy.rawrw, 'linewidth', 1.5)
title('Reweighted LASSO: Accuracy-Sparsity trade off', 'fontsize' ,14)
xlabel('Number of non-zero feautres', 'fontsize' ,14)
ylabel('Cross Validated Accuracy', 'fontsize' ,14)
subplot(1,2,2)
plot(nnzs.normrw, accuracy.normrw, 'linewidth', 1.5)


%% analysis on "best" beta 
% select the beta with best accuracy, then wit highest sparsity
a = beta.raw(:,find(accuracy.raw == max(accuracy.raw),1,'last'));
b= beta.norm(:,find(accuracy.norm == max(accuracy.norm),1,'last'));

c= beta.rawrw(:,find(accuracy.rawrw == max(accuracy.rawrw),1,'last'));
d= beta.normrw(:,find(accuracy.normrw == max(accuracy.normrw),1,'last'));

% normed difference 
norm(a-b,2)
norm(c-d,2)

% difference might be a good indicate 
numNonZeros(a) - numNonZeros(b)
numNonZeros(c) - numNonZeros(d)

% agreement about feature selection
sum(getNonZeroIdx(a) == getNonZeroIdx(b))
sum(getNonZeroIdx(c) == getNonZeroIdx(d))

%% analysis on beta with the same parameter
agreement.lasso = nan(numel(LAMDBAS),1);
normDiff.lasso = nan(numel(LAMDBAS),1);

% for regular lasso 
for i = 1 : numel(LAMDBAS)
    agreement.lasso(i) = sum(getNonZeroIdx(beta.raw(:,i)) == getNonZeroIdx(beta.norm(:,i)));
    normDiff.lasso(i) = norm(beta.raw(:,i) - beta.norm(:,i),1);
end

% for reweighted lasso 
agreement.rw = nan(numel(DELTA),1);
normDiff.rw = nan(numel(DELTA),1);
for i = 1 : numel(DELTA)
    agreement.rw(i) = sum(getNonZeroIdx(beta.rawrw(:,i)) == getNonZeroIdx(beta.normrw(:,i)));
    normDiff.rw(i) = norm(beta.rawrw(:,i) - beta.normrw(:,i),1);
end

%%

subplot(1,2,1)
barwitherr([std(agreement.lasso)], [mean(agreement.lasso)])
set(gca,'XTickLabel',{'lasso'}, 'fontsize', 15)
title('feature alignment, lasso', 'fontsize', 15)
ylim([0, size(X,2)])
ylab_text = sprintf('Feature selection "consistency" (total = %d)', size(X,2));
ylabel(ylab_text)

subplot(1,2,2)
barwitherr([std(agreement.rw)], [mean(agreement.rw)])
set(gca,'XTickLabel',{'RWlasso'}, 'fontsize', 15)
title('feature alignment, RW lasso', 'fontsize', 15)
ylim([0, size(X,2)])
