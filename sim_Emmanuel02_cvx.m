clear variables; close all; clc
% stimuli by voxel
m = 256;        % num stimuli
n = 512;        % num voxels

% generate X
X = randn(m,n);
% generate beta and y
beta.truth = generateBeta(130, n, 1);
% noise = randn(m,1);
y = X * beta.truth;

%% fitting reweighted-lasso 
[beta.rw history] = reweightedLasso_cvx(X,y);
% fit regular lasso
beta.lasso = lasso_ista(X,y,1,ones(n,1),0);


%%
numNonZeros(beta.rw)
numNonZeros(beta.lasso)

%% paper plot 
FS = 14; 
subplot(2,2,1)
plot(beta.truth, beta.rw, 'o')
title('True beta vs. estimated beta with reweighted lasso', 'fontsize', FS)
xlabel('True beta values', 'fontsize', FS)
ylabel('Estimated beta with reweighted lasso', 'fontsize', FS)
hold on 
range = min(beta.truth):0.1:max(beta.truth);
plot(range,range)
hold off


subplot(2,2,2)
% plot(beta.truth, history.beta(:,1), 'o')
plot(beta.truth, beta.lasso, 'o')
title('True beta vs. estimated beta with lasso', 'fontsize', FS)
xlabel('True beta values', 'fontsize', FS)
ylabel('Estimated beta with lasso', 'fontsize', FS)
hold on 
plot(range,range)
hold off

%%
% compute the difference between estimate vs. truth over iterations 
diff = bsxfun(@minus, history.beta, beta.truth);
% preallocate 
diffnorm = nan(size(diff,2),1);
ydev = nan(size(diff,2),1);
acc = nan(size(diff,2),1);

for i = 1 : size(diff,2)
    diffnorm(i) = norm(beta.truth-diff(:,i), 2);
    ydev(i) = norm(y - X * history.beta(:,i),2);
    acc(i) = sum(y == sign(X * history.beta(:,i)))/length(y);
end

subplot(2,2,3)
plot(diffnorm)
title('Inf-Norm of difference between estimate and truth')

subplot(2,2,4)
plot(ydev)
title('2-Norm of difference between y and X beta')