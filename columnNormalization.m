function X_normalized = columnNormalization(X)
X_normalized = nan(size(X));
for i = 1 : size(X,2); 
    X_normalized(:,i) = (X(:,i) -  mean(X(:,i))) / std(X(:,i));
end
end