clear variables; close all; clc
% stimuli by voxel
m = 256;        % num stimuli
n = 520;        % num voxels

% generate X
X = randn(m,n);
% generate beta and y
beta.truth = generateBeta(130, n);
noise = randn(m,1);
y = X * beta.truth + noise;

%% fitting reweighted-lasso 
[beta.rw history] = reweightedLasso_cvx(X,y);

%% plot
% display the final signal reconstruction alignment
subplot(1,3,1)
plot(beta.truth, beta.rw, 'o')
title('True beta vs. estimated beta')

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

subplot(1,3,2)
plot(diffnorm)
title('2-Norm of difference between estimate and truth')

subplot(1,3,3)
plot(ydev)
title('2-Norm of difference between y and X beta')

% %% plot sorted nonzero elements
% hold on
% plot(sort(beta.truth(beta.truth~=0)))       % truth
% plot(sort(beta.rw(beta.truth~=0)))          % rw lasso
% plot(sort(history.beta(beta.truth~=0,1)))   % lasso
% hold off

beta.lasso = lasso_ista(X,y,1,ones(n,1),0);

% if the truth signal have similar magnitude
[beta.truth beta.rw beta.lasso]

norm(beta.truth - beta.rw)
norm(beta.truth - beta.lasso)
