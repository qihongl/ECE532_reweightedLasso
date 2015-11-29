%% iterative soft thresholding to lasso
function [finalBeta, history] = lasso_q(X, y, lambda, tau, reWeight,display)
%% set some parameters
[m,n] = size(X);
maxIter = 1000;   % maxiteration that the program will run
tolerance = 1e-3;

%% pre-compute things that need to be computed many times
XTy = X' * y;
XTX = X' * X;
% preallocate
beta = zeros(n,1);
if display
    fprintf('iter\taccuracy\tdiff_beta\tnnz\n');
end
W = ones(n,1);
for i = 1 : maxIter;
    %% update parameter estimation 
    % iterative soft thresholding
    Z = beta(:,i) - tau * XTX * beta(:,i) +  tau * XTy;
    beta(:,i+1) = sign(Z) .* max((Z - 2 * lambda * tau * W), 0);
    if reWeight
        errors = min(beta(:,i+1)) .* rand(n,1);
        % update weights 
        W = 1 ./ (abs(beta(:,i+1)) + errors); 
    end
    %% Performance monitoring
    % compute change in beta
    diff = norm(beta(:,i+1) - beta(:,i));
    % nnz for the updated beta
    nnz = n-numZeros(beta(:,i+1));
    % error of the updated beta
    history.accuracy(i) = sum(sign(X * beta(:,i+1)) == y) / m;
    % print the performance
    if display
        fprintf('%4d%12.4f %16.4f %10d\n', i, history.accuracy(i), diff, nnz);
    end
    
    %% stopping criterion
    if diff < tolerance
        break;
    end
end

finalBeta = beta(:,end);
history.beta = beta;
% record number of non-zero weights
history.nonZeroBetas = nnz;
% plot the performance
if display
    %     plotError(1-record.accuracy)
end
end

%% function for plotting the errors
% input: error rate over iterations
function plotError(error)
FZ = 14;
plot(error, 'linewidth', 1.5)
title('Error rate over iterations', 'fontsize', FZ)
xlabel('Iterations', 'fontsize', FZ)
ylabel('Error rate', 'fontsize', FZ)
end

%% check the number of non zero betas
% input
function [nz] = numZeros(beta)
tolerance = 1e-6;
% calculate number of "zeros"
nz = sum(abs(beta) <= tolerance);
end

% input: column vector of beta
function [wts] = computeWts(beta)
wts = 1 ./ abs(beta);
end