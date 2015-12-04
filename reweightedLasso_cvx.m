function [beta, history] = reweightedLasso_cvx(X,y)
% constants
MAX_ITER = 20;
e = .05;
[~,n] = size(X); 

% initialize weights
weights = ones(n,1);

%% iteratively solving the proxy of l0 minimization
for i = 1 : MAX_ITER
    % fit contrained l1 problem with cvx
    cvx_begin quiet
        variable beta_hat(n)
        minimize(norm(weights.*beta_hat ,1))
        subject to
            X * beta_hat == y
    cvx_end
    
    % update weights
    weights = 1 ./ (abs(beta_hat) + e);
    
    % save beta
    history.beta(:,i) = beta_hat;
    
    % 
    if i >1
        diff = norm(history.beta(:,i) - history.beta(:,i-1),2);    
        fprintf('Iteration: %4d \t betaChange: %12f \n', i, diff)
        if diff < 1e-4
            break;
        end
    end
end

beta = history.beta(:,end);
end