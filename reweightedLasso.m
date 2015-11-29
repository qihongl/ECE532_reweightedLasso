%% reweighted-lasso, by Emmanuel et al.
function [beta, history] = reweightedLasso(X, y, lambda, display)
% some constants
MAX_ITER = 30;
[~,n] = size(X);

%% iteratively fitting reweighted-lasso
weights = ones(n,1);
for i = 1 : MAX_ITER
    % fit lasso
    beta = lasso_ista(X, y, lambda, weights, 0);
    % update weights
    e = .1;
    weights = 1 ./ (abs(beta) + e);
    
    %% record beta history
    history.weights(:,i) = weights;
    history.beta(:,i) = beta;
    
    if i > 1
        if display
            diff = norm(history.beta(:,i) - history.beta(:,i-1),2);
            fprintf('%d\t%f\n',i,diff);
        end
        % stopping critera
        if diff < 1e-3
            break;
        end
    end
    
end
end