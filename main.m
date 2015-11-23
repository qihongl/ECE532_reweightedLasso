clear all; close all; clc;



for a = 1: 10;
    for b = 1:10;
        % set signal and noise 
        signal.big = a;
        signal.small = a;
        noise.big = b;
        noise.small = b;
        
        % fit normalized and raw model 
        [out, finalBeta] = simNormalizedVsRaw(signal, noise);
        
        % record performance
        betaDiff(a,b) = out.diff;
        sec1OverSec2.raw(a,b) = out.raw.sec1 / out.raw.sec2;
        sec1OverSec2.norm(a,b) = out.norm.sec1 / out.norm.sec2;
        propSelected.raw(a,b) = out.raw.sec1 + out.raw.sec2;
        propSelected.norm(a,b) = out.norm.sec1 + out.norm.sec2;
    end
end


%% plot 
subplot(2,3,1)
imagesc(betaDiff)
addTexts2Plots('Beta diff norm vs. raw')
subplot(2,3,2)
imagesc(sec1OverSec2.raw); 
addTexts2Plots('Raw: numVoxels selcted: sec1 / sec2')
subplot(2,3,3)
imagesc(sec1OverSec2.norm); colorbar
addTexts2Plots('Norm: numVoxels selcted: sec1 / sec2')
subplot(2,3,5)
imagesc(propSelected.raw); colorbar
addTexts2Plots('Raw: numVoxel selected in region 1 & 2')
subplot(2,3,6)
imagesc(propSelected.norm); colorbar
addTexts2Plots('Norm: numVoxel selected in region 1 and 2')
