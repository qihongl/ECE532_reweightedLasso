%% Solve Reweighted lasso algorithm with constrained optimization 
% Note: this function has dependency on the CVX package
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% Input:    X   m by n design matrix
%           y   m by 1 response
% Output:   beta    the coefficients for the features 
%           history the beta and weights over iterations 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [beta, history] = reweightedLasso_cvx(X, y, delta)
% constants
MAX_ITER = 50;
TOLERANCE = 1e-4;

% other parameters
e = .1;
n = size(X,2);
weights = ones(n,1);

%% iteratively solving the proxy of l0 minimization
for i = 1 : MAX_ITER
    % solve contrained l1 minimization 
    cvx_begin quiet
        variable beta_hat(n)
        minimize( norm( weights .* beta_hat ,1) )   
        subject to
            norm(X * beta_hat - y) <= delta
    cvx_end
    
    % update weights
    weights = 1 ./ (abs(beta_hat) + e);
    
    % save beta and weights 
    history.beta(:,i) = beta_hat;
    history.weights (:,i) = weights; 
    
    if i > 1
        diff = norm(history.beta(:,i) - history.beta(:,i-1),2);    
        fprintf('Iteration: %4d \t betaChange: %12f \n', i, diff)
        
        % stopping criterion
        if diff < TOLERANCE
            break;
        end
    end
end
% extract final beta_estimates
beta = history.beta(:,end);
end