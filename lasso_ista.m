%% optimize lasso with iterative soft thresholding
function [beta] = lasso_ista(X, y, lambda, weights, display)
% constants
MAX_ITERS = 1e4;     % maximum number of iterations % convergence tolerance
TOLERANCE = 1e-5;
tau = .99 / norm(X)^2;  % choose stepsize
[~,n] = size(X);

beta = zeros(n,1);          % preallocate
XTX = X' * X;               % precompute matrix operations
XTy = X' * y;

%% iterative soft-thresholding
for i = 1: MAX_ITERS
    % parameter update with landweber iteration
    z = beta - tau * (XTX * beta - XTy);
    betaPrev = beta;
    beta = sign(z) .* max( abs(z) - tau * lambda * weights / 2, 0 );
    
    
    % display progress
    if display
        fprintf('Iter: %4d \t yDev: %f \t betaChange: %f\n', i, norm(y - X*beta,2), norm(beta - betaPrev));
    end
    % compute change
    if norm(beta - betaPrev) < TOLERANCE
        break
    end
    
end

end