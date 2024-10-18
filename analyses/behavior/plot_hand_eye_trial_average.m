function plot_hand_eye_trial_average

% plot ball, paddle, eye position in 2D for each condition (panels)
% color coding time 
% - ball: viridis
% - paddle: cividis
% - eye: twilight shifted

% dependence: v2struct, viridis, cividis, twilight_shifted

%% data set up
subject_id='all_hand_dmfc';
% subject_id='perle_hand_dmfc';
% subject_id='mahler_hand_dmfc';
condition='occ';
timebinsize=50;

% load data
D=load_data(subject_id,condition,timebinsize); % behavioral_responses
v2struct(D.behavioral_responses); % vis occ

% masking
mask_vis=D.masks.vis.start_occ_pad0; % [79 x 100]
mask_occ=D.masks.occ.occ_end_pad0; 

% get # conditions and time
[nCond,nT]=size(vis.ball_pos_x);
nRow=ceil(sqrt(nCond)); nCol=nRow; % try to be equal

% [roll in mask]
% start_pad0_roll0 % 49
% occ_pad0_roll-50 % 
% occ_pad0_roll49
% f_pad0_roll-50
% half_pad0
% half_pad0_roll-50
% start_end_pad0_roll-50

axis_limit=[-10 10 -10 10];
id_plot=1;

threshold=0.1; % to find bounce

%% ploting 
if id_plot
    figure;
    tiledlayout('flow','TileSpacing','none','Padding','none'); % (nRow,nCol);
end

for iCond=1:nCond
    if id_plot
        nexttile;
        %% ball
        plot_trajectory('ball_pos_x','ball_pos_y',v2struct);
        
        %% paddle
        plot_trajectory('ball_pos_x','paddle_pos_y',v2struct);
        
        %% eye
        plot_trajectory('eye_h','eye_v',v2struct);
        
        % constrain panel
        axis(axis_limit);
        % vis-occ boundary
        x_boundary=max(vis.ball_pos_x(1,mask_vis(1,:)==1));
        plotVertical(gca,x_boundary,[]);
        
    end
    
    % indicate bounce
    y0=vis.ball_pos_y(iCond,mask_vis(iCond,:)==1);
    y1=occ.ball_pos_y(iCond,mask_occ(iCond,:)==1);
    
    id_bounce_vis=sum(abs(diff(diff(y0)))>threshold);
    id_bounce_occ=sum(abs(diff(diff(y1)))>threshold);
    if id_bounce_vis
        disp([num2str(iCond) ': bounce in visible']);
    end
    if id_bounce_occ
        disp([num2str(iCond) ': bounce in occluded']);
    end
    
end % for iCond=1:nCond

set(gcf,'position',[1 1 1920 980]);

%% sub functions
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
    marker='s';
    markersize=4;
    tmpCmap=cividis(length(x0)+length(x1));
elseif contains(y,'eye')
    marker='o';
    markersize=4;
    tmpCmap=twilight_shifted(length(x0)+length(x1));
end

% visible
plotmultiple(genDiffVector(x0),genDiffVector(y0),tmpCmap(1:(length(x0)-1),:),'-',[],lw); % line
plotmultiple(x0(:),y0(:),tmpCmap(1:length(x0),:),marker,markersize,lw); % marker


% occluded
plotmultiple(genDiffVector(x1),genDiffVector(y1),tmpCmap((1+length(x0)):(end-1),:),'-',[],lw);
plotmultiple(x1(:),y1(:),tmpCmap((1+length(x0)):end,:),marker,markersize,lw);

ha;

function dx=genDiffVector(x)
% [x(1) x(2); x(2) x(3);...]
x=x(:);
dx=[x(1:(end-1)) x(2:end)];

