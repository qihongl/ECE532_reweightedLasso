%% optimize lasso with iterative soft thresholding
function [beta] = lasso_ista(X, y, lambda, weights, display)
% constants
MAX_ITERS = 1e4;     % maximum number of iterations % convergence tolerance
TOLERANCE = 1e-5;
tau = .9 / norm(X)^2;  % choose stepsize
[~,n] = size(X);

beta = zeros(n,1);          % preallocate
XTX = X' * X;               % precompute matrix operations
XTy = X' * y;

% w = ones(n,1);
w = weights; 

% iterative soft-thresholding
for i = 1: MAX_ITERS
    % parameter update with landweber iteration
    z = beta - tau * (XTX * beta - XTy);
    betaPrev = beta;
    beta = sign(z) .* max( abs(z) - tau * lambda * w / 2, 0 );
    
    % compute change
    if norm(beta - betaPrev) < TOLERANCE
        break
    end
    % display progress
    if display
        fprintf('%4d \t %f \t %f\n', i, norm(y - X*beta,2),norm(beta - betaPrev));
    end
end

end