%% get logical vector index s.t.
% true  == non-zero
% false == zero 

function [nzIdx] = getNonZeroIdx(vector)
tolerance = 1e-6;
n = length(vector);

% set all the elements to true ("non-zero") 
nzIdx = true(n,1);
% pick elements that are close to zero and said it is zero
nzIdx( abs(vector) < tolerance) = false; 

end