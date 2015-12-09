clear variables; close all; clc;

sampleSize = 10;
dev = cell(sampleSize,1);
nnzs = cell(sampleSize,1);

for i = 1 : sampleSize
    [dev{i}, nnzs{i}] = sim_Emmanuel02_cvx();
end

%% sumarize the data 
devBetaMat = nan(sampleSize,3);
devYmat = nan(sampleSize,3);
nnzMat = nan(sampleSize,3);
for i = 1 : sampleSize
    devBetaMat(i,1) = dev{i}.beta.rw;
    devBetaMat(i,2) = dev{i}.beta.lasso;
    devBetaMat(i,3) = dev{i}.beta.ls;
    devYmat(i,1) = dev{i}.y.rw;
    devYmat(i,2) = dev{i}.y.lasso;
    devYmat(i,3) = dev{i}.y.ls;
    nnzMat(i,1) = nnzs{i}.rw;
    nnzMat(i,2) = nnzs{i}.lasso;
    nnzMat(i,3) = nnzs{i}.ls;
end

%% 
subplot(1,3,1)
barwitherr([std(devBetaMat)], [mean(devBetaMat)])
set(gca,'XTickLabel',{'RW','lasso', 'ls'}, 'fontsize', 15)
title('Beta deviation', 'fontsize', 15)
ylim([0,9])

subplot(1,3,2)
barwitherr([std(devYmat)], [mean(devYmat)])
set(gca,'XTickLabel',{'RW','lasso', 'ls'}, 'fontsize', 15)
title('y deviation', 'fontsize', 15)

subplot(1,3,3)
barwitherr([std(nnzMat)], [mean(nnzMat)])
set(gca,'XTickLabel',{'RW','lasso', 'ls'}, 'fontsize', 15)
ylabel('Number of non-zero features', 'fontsize', 15)
title('Sparsity', 'fontsize', 15)
