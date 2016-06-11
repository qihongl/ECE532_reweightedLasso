function beta = lasso_solve( X, y, lambda, display ) %ista_solve: Iterative soft-thresholding!
% % %
% this function solves the minimization problem
% Minimize |Ax-y|_2^2 + lambda*|beta|_1 (Lasso regression) using iterative soft-thresholding.
MAX_ITERS = 1e5; 
TOLERANCE = 1e-3;

tau = 1/norm(X)^2;
[~,n] = size(X); 

XTX = X' * X;               
XTy = X' * y;
% 
% % maximum number of iterations % convergence tolerance
% % choose stepsize
% % start point for the iteration
% for i = 1: MAX_ITER
%     z = beta - tau*(X'*(X*beta-y));  % Landweber 
%     xold = beta;  % store old beta 
%     beta = sign(z) .* max( abs(z) - tau*lambda/2, 0 );  % ISTA
%     if norm(beta-xold) < TOL
%         break
%     end
% end

%% iterative soft-thresholding
beta = zeros(n,1);          % preallocate
for i = 1: MAX_ITERS
    % parameter update with landweber iteration
    z = beta - tau * (XTX * beta - XTy);
    betaPrev = beta;
    beta = sign(z) .* max( abs(z) - tau * lambda / 2, 0 );
    
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