function plotPerformance(beta_truth, history, X, y)
FS = 16; 

%% plot the comparison between estimated beta versus the true beta
% reweighted estimation 
subplot(2,2,1)
plot(beta_truth, history.beta(:,end), 'o')
title('True beta vs. estimated beta, reweighted lasso', 'fontsize', FS)
xlabel('$ \mathbf{ \beta_{truth}} $','Interpreter','LaTex', 'fontsize', FS)
ylabel('$ \mathbf{ \hat{\beta_{RW}}} $','Interpreter','LaTex', 'fontsize', FS)
hold on 
range = min(beta_truth):0.1:max(beta_truth);
plot(range,range)
hold off

% regular lasso estimation 
subplot(2,2,2)
plot(beta_truth, history.beta(:,1), 'o')
title('True beta vs. estimated beta, lasso', 'fontsize', FS)
xlabel('$\mathbf{ \beta_{truth}}$','Interpreter','LaTex', 'fontsize', FS)
ylabel('$\mathbf{ \hat{\beta_{lasso}}} $','Interpreter','LaTex', 'fontsize', FS)
hold on 
plot(range,range)
hold off


%% For reweighted lasso, compute the performance over iterations  
diff = bsxfun(@minus, history.beta, beta_truth);
% preallocate 
diffnorm = nan(size(diff,2),1);
ydev = nan(size(diff,2),1);
acc = nan(size(diff,2),1);

for i = 1 : size(diff,2)
    diffnorm(i) = norm(beta_truth-diff(:,i), 2);
    ydev(i) = norm(y - X * history.beta(:,i),2);
    acc(i) = sum(y == sign(X * history.beta(:,i)))/length(y);
end

% plot the Difference between beta and estimated beta over iterations 
subplot(2,2,3)
plot(diffnorm, 'linewidth',1.5)
title('Difference between beta and estimated beta', 'fontsize', FS)
ylabel('$\mathbf{\| \beta_{truth} - \hat{\beta_{RW}} \|_2 }$','Interpreter','LaTex', 'fontsize', FS)
xlabel('Iterations', 'fontsize', FS)

% plot the deviation of estimated response over iterations 
subplot(2,2,4)
plot(ydev, 'linewidth',1.5)
ylabel('$\mathbf{\| y_{truth} - X \hat{\beta_{RW}} \|_2 }$','Interpreter','LaTex', 'fontsize', FS)
title('Difference between true y and predicted y', 'fontsize', FS)

xlabel('Iterations', 'fontsize', FS)
end