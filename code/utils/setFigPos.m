function setFigPos(i,j)
drawnow;
load pplot.mat;
pause(0.01);
if j<0
    set(gcf,'position',pplot.(['rect' num2str(i) '_m' num2str(-j)]));
else
    set(gcf,'position',pplot.(['rect' num2str(i) '_' num2str(j)]));
end
  