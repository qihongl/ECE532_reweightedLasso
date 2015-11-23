function [ ] = addTexts2Plots( title_text )
%ADDTEXTS2PLOTS Summary of this function goes here
%   Detailed explanation goes here
colormap jet
colorbar
FS = 14;

title(title_text, 'fontsize',FS)
xlabel('noise magnitude', 'fontsize',FS)
ylabel('signal magnitude', 'fontsize',FS)

end

