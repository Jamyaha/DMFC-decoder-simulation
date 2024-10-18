function figure1c

% plot eye/hand error as a function of time (before/after occlusion) across all conditions
% eye error = rmse(eyeXY,ballXY), rmse(eyeXY,targetXY)
% hand error = rmse(paddleY,targetY) - note that paddleX doesn't matter & paddleY0 =0

% figure 1, example trial - 2D (plot error vector)
% figure 2, averages

% dependence: v2struct

% 2023/8/29
% get task performance

% 2024/10/9: id_print_value

%% data set up
% subject_id='perle_hand_dmfc';
% subject_id='mahler_hand_dmfc';
subject_id='all_hand_dmfc';
condition='occ';
timebinsize=50;

% load data
D=load_data(subject_id,condition,timebinsize); % behavioral_responses
v2struct(D.behavioral_responses); % vis occ id_bounce

% masking
mask_vis=D.masks.vis.start_occ_pad0; % [79 x 100]
mask_occ=D.masks.occ.occ_end_pad0; 

% get # conditions and time
[nCond,nT]=size(vis.ball_pos_x);
nRow=ceil(sqrt(nCond)); nCol=nRow; % try to be equal

id_print_value=1;

%% get task performance % 2023/8/29
% find last time bin
last_time_bin=[];
[I,J]=find(~isnan(mask_occ));
for i=1:size(mask_occ,1)
    last_time_bin=[last_time_bin; max(J(I==i))];
end

% compute MAE
abs_err=[];
for i=1:size(mask_occ,1)
    ball_y=occ.ball_pos_y_TRUE(i,last_time_bin(i));
    paddle_y=occ.paddle_pos_y(i,last_time_bin(i));
    abs_err=[abs_err; abs(ball_y-paddle_y)];
end

% save_direct='/Users/hansem/Dropbox (MIT)/MPong/phys/results/rnn_comparison_results';
% save(fullfile(save_direct,[subject_id '_behavior_error.mat']),'abs_err');

%% 
% plot options
axis_limit=[-10 10 -10 10];
id_plot_all=0; % 1;
i_plot=1;

% plotting conditions
if id_plot_all
    condPlot=1:nCond;
    figure;
    tiledlayout('flow','TileSpacing','none','Padding','none'); % (nRow,nCol);
else
    nCondPlot=3; % 6;
    list_cond=[25; 74; 52]; % chosen for eg figures; randperm(nCond);
    condPlot=list_cond(1:nCondPlot);
end

%% ploting 
for iCond=1:nCond
    if id_plot_all
        nexttile;
    else
        if sum(iCond==condPlot)
            figure;
            setFigPos(floor((i_plot-1)/3)+1,mod(i_plot-1,3)+1);
            i_plot=i_plot+1;
        end
    end
    
    [eye_ball_vis{iCond},eye_ball_occ{iCond}]=get_error('eye','ball',v2struct);
    [eye_target_vis{iCond},eye_target_occ{iCond}]=get_error('eye','target',v2struct);
    [paddle_target_vis{iCond},paddle_target_occ{iCond}]=get_error('paddle','target',v2struct);
        
    if id_plot_all
        plot_error('eye','ball',v2struct);
    elseif sum(iCond==condPlot)
        plot_error('eye','ball',v2struct);
        plot_trajectory('ball_pos_x','paddle_pos_y',v2struct);
        applytofig4paper;
    end
end % for iCond=1:nCond

if id_plot_all
    set(gcf,'position',[1 1 1920 980]);
end

% get shuffled data

% but how to deal with different legnth? get shorter with attrition?

%% averaging
tmpCmap=[1 0 0; 0 0 1]; % cividis(4);
lw=2;

% eye vs ball
i_cmap=1;
hf{1}=figure; setFigPos(1,-2); % visible
id_lock=1; 
plot_average(eye_ball_vis,v2struct,'eye_ball_vis');

hf{2}=figure; setFigPos(1,-1); % occlusion
id_lock=0;
plot_average(eye_ball_occ,v2struct,'eye_ball_occ');

% match_ylim(hf(1:2));

% % eye vs target
% i_cmap=2;
% hf{3}=figure; setFigPos(2,-2); % visible
% id_lock=1; 
% plot_average(eye_target_vis,v2struct);
% 
% hf{4}=figure; setFigPos(2,-1); % occlusion
% id_lock=0;
% plot_average(eye_target_occ,v2struct);

% match_ylim(hf(3:4));

% paddle vs target
i_cmap=2; % 3;
% hf{5}=figure; setFigPos(2,1); % visible
figure(hf{1});
id_lock=1; 
plot_average(paddle_target_vis,v2struct,'paddle_target_vis');

% hf{6}=figure; setFigPos(2,2); % occlusion
figure(hf{2});
id_lock=0;
plot_average(paddle_target_occ,v2struct,'paddle_target_occ');

% match_ylim(hf(5:6));

%% sub functions
function plot_average(x,struct,y)
v2struct(struct);
[mu,sigma]=meanCell(x,id_lock);
if id_lock==0
    tmpX=0:timebinsize:((length(mu)-1)*timebinsize);
elseif id_lock==1
    tmpX=-((length(mu)-1)*timebinsize):timebinsize:0;
end
% individual conditions
tmp_mat=fillNanCell(x,id_lock); % [#Condition x time]
% plot(tmpX,tmp_mat,'-','linewidth',1,'color',[.5 .5 .5]);
% averge
shadedErrorBar(tmpX,mu,sigma,{'-','color',tmpCmap(i_cmap,:),'linewidth',lw},1); ha;
% axes
axis tight;
ymin=0; 
ymax=10;
ylim([ymin ymax]);
applytofig4paper;

if id_print_value
    if contains(y,'eye_ball_vis')
        disp('-----eye_ball_vis-----');
        disp([tmpX(:)]); disp([mu(:) sigma(:)]);
    elseif contains(y,'eye_ball_occ')
        disp('-----eye_ball_occ-----');
        disp([tmpX(:)]); disp([mu(:) sigma(:)]);
    elseif contains(y,'paddle_target_vis')
        disp('-----paddle_target_vis-----');
        disp([tmpX(:)]); disp([mu(:) sigma(:)]);
    elseif contains(y,'paddle_target_occ')
        disp('-----paddle_target_occ-----');
        disp([tmpX(:)]); disp([mu(:) sigma(:)]);
        xlim([0 1500]);
    end
end

function [error_vis,error_occ]=get_error(xx,yy,struct)
%%
% get data
v2struct(struct);
if contains(xx,'eye')
    x=get_data('eye_h','eye_v',struct); % x0(visible) y0 x1(occluded) y1
elseif contains(xx,'paddle')
    x=get_data('ball_pos_x','paddle_pos_y',struct); % paddle_pos_x doesn't exist
end
if contains(yy,'ball')
    y=get_data('ball_pos_x','ball_pos_y',struct);
elseif contains(yy,'target')
    y=get_data('ball_pos_x','ball_pos_y',struct);
    y.y0(1:end)=y.y1(end); % even for visible, use final target position in occluded
    y.y1(1:end)=y.y1(end);
    y.x0=x.x0; % to ignore x
    y.x1=x.x1; % to ignore x
end
% error
error_vis=sqrt((x.x0-y.x0).^2+(x.y0-y.y0).^2);
error_occ=sqrt((x.x1-y.x1).^2+(x.y1-y.y1).^2);
% convert to row vector
error_vis=error_vis(:)';
error_occ=error_occ(:)';

function plot_error(xx,yy,struct)
%%
% get data
v2struct(struct);
if contains(xx,'eye')
    x=get_data('eye_h','eye_v',struct); % x0 y0 x1 y1
elseif contains(xx,'paddle')
    x=get_data('ball_pos_x','paddle_pos_y',struct); % paddle_pos_x doesn't exist
end
if contains(yy,'ball')
    y=get_data('ball_pos_x','ball_pos_y',struct);
elseif contains(yy,'target')
    y=get_data('ball_pos_x','ball_pos_y',struct);
    y.y0(1:end)=y.y1(end);
    y.y1(1:end)=y.y1(end);
    y.x0=x.x0; % to ignore x
    y.x1=x.x1; % to ignore x
end
% condition specific plot options
lw=2;
markersize=4;
if contains(xx,'eye')
    marker='o';
    tmpCmap=viridis(length(x.x0)+length(x.x1));
elseif contains(xx,'paddle')
    marker='s';
    tmpCmap=twilight_shifted(length(x.x0)+length(x.x1));
end
    
% plot
if id_plot_all || sum(iCond==condPlot)
    
    % x: eye or paddle
    plotmultiple(genDiffVector(x.x0),genDiffVector(x.y0),zeros(length(x.x0),3),'-',[],[]); % visible
    plotmultiple(genDiffVector(x.x1),genDiffVector(x.y1),zeros(length(x.x1),3),'-',[],[]); % occluded
    
    % y: ball or target
    plotmultiple(genDiffVector(y.x0),genDiffVector(y.y0),zeros(length(y.x0),3),'-',[],[]); % visible
    plotmultiple(genDiffVector(y.x1),genDiffVector(y.y1),zeros(length(y.x1),3),'-',[],[]); % occluded
        
    % error vector
    tmpX=[x.x0(:) y.x0(:); x.x1(:) y.x1(:)];
    tmpY=[x.y0(:) y.y0(:); x.y1(:) y.y1(:)];
    plotmultiple(tmpX,tmpY,tmpCmap,'-',[],lw); % line
    plotmultiple(tmpX(:,1),tmpY(:,1),tmpCmap,'o',markersize,lw); % marker

    if id_print_value
        disp('----- BALL ------');
        disp([y.x0(:) y.y0(:); y.x1(:) y.y1(:)]);
        disp('----- EYE ------');
        disp([x.x0(:) x.y0(:); x.x1(:) x.y1(:)]);
    end
    
    ha;
    
    % constrain panel
    axis(axis_limit);
    % vis-occ boundary
    x_boundary=max(vis.ball_pos_x(1,mask_vis(1,:)==1));
    plotVertical(gca,x_boundary,[]);
    
end

function D=get_data(x,y,struct)

v2struct(struct);
x0=vis.(x)(iCond,mask_vis(iCond,:)==1);% visible
y0=vis.(y)(iCond,mask_vis(iCond,:)==1);
x1=occ.(x)(iCond,mask_occ(iCond,:)==1);% occluded
y1=occ.(y)(iCond,mask_occ(iCond,:)==1);

D=v2struct(x0,y0,x1,y1);

function plot_trajectory(x,y,struct)

% x0, y0 for visible; x1, y1 for occluded

%%
% get data
v2struct(struct);
x0=vis.(x)(iCond,mask_vis(iCond,:)==1);% visible
y0=vis.(y)(iCond,mask_vis(iCond,:)==1);
x1=occ.(x)(iCond,mask_occ(iCond,:)==1);% occluded
y1=occ.(y)(iCond,mask_occ(iCond,:)==1);

% condition specific plot options
lw=2;
if contains(y,'ball')
    marker='.';
    markersize=1;
    tmpCmap=viridis(length(x0)+length(x1));
elseif contains(y,'paddle')
    marker='.'; % 's';
    markersize=4;
%     tmpCmap=cividis(length(x0)+length(x1));

    % determine color based on y position (consistent with paper's colormap)
    cmap0=getColor(y0);
    cmap1=getColor(y1);

elseif contains(y,'eye')
    marker='o';
    markersize=4;
    tmpCmap=twilight_shifted(length(x0)+length(x1));
end

% visible
plotmultiple(genDiffVector(x0),genDiffVector(y0),cmap0,'-',[],lw); % line
plotmultiple(x0(:),y0(:),cmap0,marker,markersize,lw); % marker
% plotmultiple(genDiffVector(x0),genDiffVector(y0),tmpCmap(1:(length(x0)-1),:),'-',[],lw); % line
% plotmultiple(x0(:),y0(:),tmpCmap(1:length(x0),:),marker,markersize,lw); % marker

% occluded
plotmultiple(genDiffVector(x1),genDiffVector(y1),cmap1,'-',[],lw);
plotmultiple(x1(:),y1(:),cmap1,marker,markersize,lw);
% plotmultiple(genDiffVector(x1),genDiffVector(y1),tmpCmap((1+length(x0)):(end-1),:),'-',[],lw);
% plotmultiple(x1(:),y1(:),tmpCmap((1+length(x0)):end,:),marker,markersize,lw);

if id_print_value
    disp('----- PADDLE ------');
    disp([x0(:) y0(:); x1(:) y1(:)]);
end

ha;

function cmap_out=getColor(y)
ymin=-10;
ymax=10;
dy=0.01; % min(abs(diff(y)));

n=round((ymax-ymin)/dy);
tmpCmap=plasma(n);
cmap_out=nan(length(y),3);
for i=1:length(y)
    index=find_index(y(i),linspace(ymin,ymax,n));
    cmap_out(i,:)=tmpCmap(index,:);
end

function index=find_index(y,x)
distance=abs(y-x);
index=find(min(distance)==distance,1);

function dx=genDiffVector(x)
% [x(1) x(2); x(2) x(3);...]
x=x(:);
dx=[x(1:(end-1)) x(2:end)];