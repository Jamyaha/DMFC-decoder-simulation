function applytofig4paper

box off;
set(gca,'tickdir','out');
set(gcf,'Units','centimeters');
load pplot.mat;
% load('/Users/hansem/Dropbox (MIT)/timeArithmetic/pplot.mat'); % optsExpFig4paper
applytofig(gcf,optsExpFig4paper);

%%
% optsExpFig4paper=optsExpFig;
% optsExpFig4paper.Format='eps';
% optsExpFig4paper.Width=3;
% optsExpFig4paper.Height=3;
% optsExpFig4paper.DefaultFixedLineWidth=0.5;
% optsExpFig4paper.Renderer='painters';
% optsExpFig4paper.Bounds='loose';
% optsExpFig4paper.FontMode='fixed';
% optsExpFig4paper.DefaultFixedFontSize=7;
% optsExpFig4paper.FontSize=7;
% optsExpFig4paper.LineWidth=0.5;
% optsExpFig4paper.Resolution=600;
% optsExpFig4paper.LockAxes='on';

%% 
% /Users/hansem/Dropbox (MIT)/misc/MATLABpref_backup/jazlab2iMac_20180621/ExportSetup/RSGpr_lwp5.txt

% Version 1
% Format eps
% Preview none
% Width 3
% Height 3
% Units centimeters
% Color rgb
% Background w
% FixedFontSize 10
% ScaledFontSize auto
% FontMode scaled
% FontSizeMin 8
% FixedLineWidth 0.5
% ScaledLineWidth auto
% LineMode fixed
% LineWidthMin 0.5
% FontName auto
% FontWeight auto
% FontAngle auto
% FontEncoding latin1
% PSLevel 3
% Renderer painters
% Resolution 600
% LineStyleMap none
% ApplyStyle 0
% Bounds loose
% LockAxes on
% LockAxesTicks off
% ShowUI on
% SeparateText off

