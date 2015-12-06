%% reweighted-lasso
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% input     X           m by n design matrix
%           y           m by 1 responses 
%           display     show progress
% output    beta        the lasso solution
%           history     beta & weights over iterations 
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
function [beta, history] = reweightedLasso(X, y, display)
% some constants
MAX_ITER = 30;
[~,n] = size(X);
tolerance = 1e-6;

%% iteratively fitting reweighted-lasso
weights = ones(n,1);
for i = 1 : MAX_ITER
    % fit lasso
    beta = lasso_ista(X, y, 1, weights, 0);
    % update weights
    e = .1;
    weights = 1 ./ (abs(beta) + e);
    
    %% record beta history
    history.weights(:,i) = weights;
    history.beta(:,i) = beta;
    
    if i > 1
        diff = norm(history.beta(:,i) - history.beta(:,i-1),2);
        % stopping critera
        if diff < tolerance
            break;
        end
        if display  
            fprintf('%d\t%f\n',i,diff);
        end
    end
    
end
end