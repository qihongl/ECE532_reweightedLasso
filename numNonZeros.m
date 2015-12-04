function [nnzs] = numNonZeros(vector)
tolerance = 1e-6;

nnzs = length(vector) - sum(abs(vector) < tolerance);


end