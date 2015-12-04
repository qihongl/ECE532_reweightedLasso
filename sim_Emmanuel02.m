clear variables; close all; clc
% stimuli by voxel
m = 256;        % num stimuli
n = 520;        % num voxels
rng(1)
% generate X
X = randn(m,n);

% generate beta and y 
beta.truth = generateBeta(100, n);
beta.truth(beta.truth~=0) = 1;

y = X * beta.truth;

%% iteratively fitting reweighted-lasso
lambda = 1;
[beta.rw, history] = reweightedLasso(X, y, lambda, true);


%% plot 
% display the final signal reconstruction alignment
subplot(1,3,1)
plot(beta.truth, history.beta(:,end), 'o')
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
title('Inf-Norm of difference between estimate and truth')

subplot(1,3,3)
plot(ydev)
title('2-Norm of difference between y and X beta')

% %% plot sorted nonzero elements 
% hold on 
% plot(sort(beta.truth(beta.truth~=0)))       % truth
% plot(sort(beta.rw(beta.truth~=0)))          % rw lasso
% plot(sort(history.beta(beta.truth~=0,1)))   % lasso
% hold off

% if the truth signal have similar magnitude 
[beta.truth beta.rw history.beta(:,1)]
