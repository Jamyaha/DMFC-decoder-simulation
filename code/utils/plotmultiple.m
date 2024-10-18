function h=plotmultiple(x,y,cmap,linespec,markersize,linewidth)

% function h=plotmultiple(x,y,cmap,linespec,markersize)
% input 
%       x: (n or 1) x m matrix (can be empty)
%       y: n x m matrix
%       cmap: n x 3 matrix (can be empty; then use jet)
%       linespec: n x m matrix (can be empty; then use '-')
%       markersize: can be empty (otherwise, 8)
%       linewidth: can be empty (otherwise, 1)
% output
%       h: handle matrix [n x 1]
%
% 2019/7/8: plot random index
% 2021/2/4: for '-', connect consecutive datapoints
% 2022/4/6: undo above 

ha;
if isempty(cmap)
    cmap=jet(size(y,1));
end
if isempty(linespec)
    linespec='-';
end
if isempty(markersize)
    markersize=8;
end
if isempty(linewidth)
    linewidth=1;
end
if ~isempty(x)
    n=size(y,1);
    if size(x,1)~=size(y,1)
        x=repmat(x,size(y,1),1);
    end
    h=nan(n,1);
    
    if strcmp(linespec,'-')
        for i=1:n
            h(i)=plot(x((i),:),y((i),:),linespec,'color',cmap((i),:),'markersize',markersize,'linewidth',linewidth,'markerfacecolor','w');
%             h(i:i+1)=plot(x(i:(i+1),:),y(i:(i+1),:),linespec,'color',cmap(i,:),'markersize',markersize,'linewidth',linewidth,'markerfacecolor','w');
        end        
    else    
        id=randperm(n);
        for i=1:n
            h(i)=plot(x(id(i),:),y(id(i),:),linespec,'color',cmap(id(i),:),'markersize',markersize,'linewidth',linewidth,'markerfacecolor','w');
        end
    end % strcmp(linespec,'-')
else
    n=size(y,1);
    h=nan(n,1);
    id=randperm(n);
    for i=1:n
        h(i)=plot(y(id(i),:),linespec,'color',cmap(id(i),:),'linewidth',linewidth);
    end
end

