clear variables; close all; clc
% stimuli by voxel
m = 256;        % num stimuli
n = 520;        % num voxels
% generate X
X = randn(m,n);

% generate beta and y 
beta.truth = generateBeta(100, n);
beta.truth(beta.truth~=0) = 1;
noise = randn(m,1);
y = X * beta.truth + noise;

%% Compute Cost and Gradient using fminunc
% Initialize fitting parameters and options
initial_beta = zeros(n, 1);
options = optimset('GradObj', 'on', 'MaxIter', 400);
%  Run fminunc to obtain the optimal beta
beta.logistic = fminunc(@(t)(costFunction(t, X, y)), initial_beta, options);
