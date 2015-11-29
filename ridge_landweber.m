%% iterative soft thresholding to lasso
function [beta, record] = ridge_landweber(X, y, lambda, tau, display)
%% set some parameters
maxIter = 10000;   % maxiteration that the program will run
tolerance = 1e-3;

%% compute things that need to be computed many times
[m,n] = size(X);
tau_XTy = tau * X' * y;
tau_XTX = tau * X' * X;
% preallocate
beta = zeros(n,1);
if display
    fprintf('iter\taccuracy\tdiff_beta\tnnz\n');
end
for i = 1 : maxIter;
    %% update weight
    beta(:,i+1) = (beta(:,i) - tau_XTX * beta(:,i) + tau_XTy) / (1 + tau * lambda);
    
    %% Performance monitoring
    % compute change in beta
    diff = norm(beta(:,i+1) - beta(:,i));
    % nnz for the updated beta
    nnz = n-numZeros(beta(:,i+1));
    % error of the updated beta
    record.accuracy(i) = sum(sign(X * beta(:,i+1)) == y) / m;
    % print the performance
    if display
        fprintf('%4d%12.4f %16.4f %10d\n', i, record.accuracy(i), diff, nnz);
    end
    
    %% stopping criterion
    if diff < tolerance
        break;
    end
end

% record number of non-zero weights
record.nonZeroBetas = nnz;
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