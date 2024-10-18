function plot_eg_glm_fit(eg_neuron_id,varargin)

% GLM
% independent variables: mean firing rate (trial-average, across-time, across-condition)
% dependent variables: ball, hand, eye-related variables
%
% dependence: v2struct, viridis, cividis, twilight_shifted
%
% goal: % of neurons selective for dependent variables
% separately for vis vs occ
% cross validation
%
% Questions
% - how to deal with latency?
% - reliable only?
% - why ball_pos_x_TRUE vs ball_pos_x in behavioral_responses.vis.ball_pos_x_TRUE
% - "vis" "occ" does not seem to reflect which event time-locked; what's difference?
% 
% TODO
% - deal with model complexity

% 2023/5/6: re-run with all data (before only perle) + aicbic + debug iModel for occ
% 2023/5/14: CV partioning persistent across models (probably not necessary though)
% 2023/5/27: PVAF 1-var(STATS.resid)./var(Y)

% 2023/7/3: plot predicted vs true FR for example neurons (changes
% indicated by %&%; originally from run_glm.m)

%% data set up
subject_id='all_hand_dmfc'; % vis 1385 occ 2576
% subject_id='perle_hand_dmfc'; % vis 1302 occ 2058
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

% delay parameters
p_delay.range_bin=-6:1:6; % -10:1:10; % -10:0:10; % 6 ; % 15; % timebinsize=50;

% - reliable only?
varname_indep='neural_responses';   % neural_responses_reliable

%% preprocessing
% create non-existant variables
[vis,occ]=create_variables(vis,occ);

% DEPENDENT variable set
fname_vis={... % ball
    'ball_pos_x_TRUE','ball_pos_y_TRUE',...
    'ball_pos_dx_TRUE','ball_pos_dy_TRUE',...
    'ball_pos_dtheta_TRUE','ball_pos_dspeed_TRUE',...
    'paddle_pos_y',... % hand
    'joy',...                           % ~dy_paddle or hand_position
    'eye_h','eye_v',... % eye
    'eye_dh','eye_dv',...
    'eye_dtheta','eye_dspeed',...
    'id_bounce_vis',...  % bounce
    };
n_fname_vis=length(fname_vis);
fname_occ={... % ball
    'ball_pos_x_TRUE','ball_pos_y_TRUE',...
    'ball_pos_dx_TRUE','ball_pos_dy_TRUE',...
    'ball_pos_dtheta_TRUE','ball_pos_dspeed_TRUE',...
    'paddle_pos_y',... % hand
    'joy',...               % ~dy_paddle or hand_position
    'eye_h','eye_v',...  % eye
    'eye_dh','eye_dv',...
    'eye_dtheta','eye_dspeed',...
    'id_bounce_occ',...  % bounce
    };
n_fname_occ=length(fname_occ);
[V,O]=generate_design_matrix(vis,occ,fname_vis,fname_occ); % [15var x 79 x 100]

% disp(fname_vis);
% show_movie(V,1,inf);
% disp(fname_occ);
% show_movie(O,1,inf);

% dim_tmp=1; id_wait=1;
% show_movie(V,dim_tmp,id_wait);

% vectorize masks
id_mask_vis=reshape(mask_vis,1,numel(mask_vis)); % [1× (79×100) double]
id_mask_occ=reshape(mask_occ,1,numel(mask_occ)); % [1× (79×100) double]

% reshape dependent variables
V=reshape_firing_rate(V,id_mask_vis); % [(79 x 100) x 15var]' > [15var x 2040]
O=reshape_firing_rate(O,id_mask_occ); % [15var x 1541]

% INDEPENDENT variable set
r_vis=reshape_firing_rate(D.(varname_indep).vis,id_mask_vis);
r_occ=reshape_firing_rate(D.(varname_indep).occ,id_mask_occ);
% D.neural_responses_reliable.occ % [1889×79×100 double]
%     D.reliable_neural_idx % occ: [1×1889 int64]
% D.neural_responses
%     occ: [2576×79×100 double]
%     vis: [1385×79×100 double]

% get neuron id %&%
idx_occ=D.neural_idx_global.occ.neural_responses;
idx_vis=D.neural_idx_global.vis.neural_responses;
% figure plot options %&%
hFig=figure; hold all;

%% dealing with delays 

if isempty(varargin)
    % load saved %&%
    load run_glm.mat dim_time LL_vis Dev_vis LL_occ Dev_occ;
else
    if strcmp(varargin{1},'egocentric')
        load run_glm_egocentric_ball.mat dim_time LL_vis Dev_vis LL_occ Dev_occ;
    elseif strcmp(varargin{1},'allocentric')
        load run_glm.mat dim_time LL_vis Dev_vis LL_occ Dev_occ;
    end
end

% dim_time=3;
% LL_vis=nan(size(r_vis,1),length(p_delay.range_bin)); Dev_vis=nan(size(LL_vis));
% LL_occ=nan(size(r_occ,1),length(p_delay.range_bin)); Dev_occ=nan(size(LL_occ));
% for iBin=1:length(p_delay.range_bin)
%     nBin=p_delay.range_bin(iBin); % 0:10
%     
%     r_vis_shift=circshift(D.(varname_indep).vis,-nBin,dim_time); % bring later neural data to early
%     r_vis=reshape_firing_rate(r_vis_shift,id_mask_vis);
%     
%     r_occ_shift=circshift(D.(varname_indep).occ,-nBin,dim_time); % bring later neural data to early
%     r_occ=reshape_firing_rate(r_occ_shift,id_mask_occ);
%     
%     nr_vis=r_vis*timebinsize; % convert back to # sp
%     nr_occ=r_occ*timebinsize;
%     
%     for iNeuron=1:size(r_vis,1)
%         [Dev_vis(iNeuron,iBin),LL_vis(iNeuron,iBin)]=run_glm_fit(nr_vis(iNeuron,:)',V');
%     end
%     for iNeuron=1:size(r_occ,1)
%         [Dev_occ(iNeuron,iBin),LL_occ(iNeuron,iBin)]=run_glm_fit(nr_occ(iNeuron,:)',O');
%     end
% end

% find maximum time points
[~,tMax_vis]=max(LL_vis,[],2);
[~,tMax_occ]=max(LL_occ,[],2);

%% conclusion: best delay

%% do nested (multiple) linear regression
% set up indices
id_pos=...
    [1 1 zeros(1,n_fname_vis-2)];
id_kine=...
    [0 0 1 1 1 1 zeros(1,n_fname_vis-6)];
id_hand=...
    [0 0 0 0 0 0 ones(1,2) zeros(1,n_fname_vis-8)];
id_eye=...
    [0 0 0 0 0 0 0 0 ones(1,6) zeros(1,n_fname_vis-14)];
id_ball=...
    [1 1 1 1 1 1 zeros(1,n_fname_vis-6)];
id_mat=logical([ones(1,n_fname_vis)]);% ... % full %&%
%     1-id_pos; ... % w/o
%     1-id_kine; ...
%     1-id_hand; ...
%     1-id_eye; ...
%     1-id_ball;...
%     id_pos; ... % w/ 7th
%     id_kine; ... 
%     id_hand; ...
%     id_eye; ...
%     id_ball]);

nModel=size(id_mat,1); % 11 % 1+4+4; % full+w/o+w/

Dev_model_vis=nan(size(r_vis,1),nModel); LL_model_vis=nan(size(Dev_model_vis));
Dev_model_occ=nan(size(r_occ,1),nModel); LL_model_occ=nan(size(Dev_model_occ));

infoCrit_model_vis=cell(size(LL_model_vis));
infoCrit_model_occ=cell(size(LL_model_occ));

PVAF_model_vis=nan(size(Dev_model_vis));
PVAF_model_occ=nan(size(Dev_model_occ));

disp('===== vis =====');

id_vis_occ=1; %&%
for iNeuron=1:size(r_vis,1)
    % select neurons %&%
    if eg_neuron_id==idx_vis(iNeuron)
        
        nBinOptimal=p_delay.range_bin(tMax_vis(iNeuron));
        
        r_vis_shift=circshift(D.(varname_indep).vis,-nBinOptimal,dim_time); % bring later neural data to early
        r_vis=reshape_firing_rate(r_vis_shift,id_mask_vis);
        
        nr_vis=r_vis*timebinsize; % convert back to # sp
        
        tmpY=nr_vis(iNeuron,:)';
        
        % persistent CV
        tmpC = cvpartition(length(tmpY),'kfold',10);
        
        for iModel=1:nModel
            tmpX=V(id_mat(iModel,:),:)';
            [Dev_model_vis(iNeuron,iModel),LL_model_vis(iNeuron,iModel),PVAF_model_vis(iNeuron,iModel)]=run_glm_fit(tmpY,tmpX,tmpC,id_vis_occ);
            % aicbic
            nObs=size(nr_vis,2);
            nParams=sum(id_mat(iModel,:))+1; % constant
            [~,~,infoCrit_model_vis{iNeuron,iModel}]=aicbic(LL_model_vis(iNeuron,iModel),nParams,nObs);
        end
        
    end % if eg_neuron_id==idx_vis(iNeuron)
end

disp('===== occ =====');

id_vis_occ=2; %&%
for iNeuron=1:size(r_occ,1)
    % select neurons %&%
    if eg_neuron_id==idx_occ(iNeuron)
        
        nBinOptimal=p_delay.range_bin(tMax_occ(iNeuron));
        
        r_occ_shift=circshift(D.(varname_indep).occ,-nBinOptimal,dim_time); % bring later neural data to early
        r_occ=reshape_firing_rate(r_occ_shift,id_mask_occ);
        
        nr_occ=r_occ*timebinsize;
        
        tmpY=nr_occ(iNeuron,:)';
        
        % persistent CV
        tmpC = cvpartition(length(tmpY),'kfold',10);
        
        for iModel=1:nModel
            tmpX=O(id_mat(iModel,:),:)';
            [Dev_model_occ(iNeuron,iModel),LL_model_occ(iNeuron,iModel),PVAF_model_occ(iNeuron,iModel)]=run_glm_fit(tmpY,tmpX,tmpC,id_vis_occ);
            % aicbic
            nObs=size(nr_occ,2);
            nParams=sum(id_mat(iModel,:))+1; % constant
            [~,~,infoCrit_model_occ{iNeuron,iModel}]=aicbic(LL_model_occ(iNeuron,iModel),nParams,nObs);
        end
        
    end %     if eg_neuron_id==idx_occ(iNeuron)
end

% save run_glm.mat; % %&%

%% subfunctions
%%
function [DEV,LL,PVAF]=run_glm_fit(Y,X,varargin)
%%
% 2023/5/4: poisson regression is only for single trials; mean firing rate ~ Normal
% 2023/5/14: varargin for persistent CV partition

% plot options
cmap=viridis(2); %&%

% CV
if ~isempty(varargin)
    c=varargin{1};
    id_vis_occ=varargin{2}; %&%
else
    c = cvpartition(length(Y),'kfold',10);
end

LL=0;
for i=1:c.NumTestSets
    [B,DEV,glmStat]=glmfit(X(training(c,i),:),Y(training(c,i)),'normal'); % ,'poisson'); % ,'constant','on'); % glmStat.p
    nHat=glmval(B,X(test(c,i),:),'identity'); % ,'log'); %
    % if rand<0.05
    %     figure; scatterhist(nHat,Y); plotIdentity(gca); waitforbuttonpress; close;
    % end
    LL=LL+nansum(log(normpdf(Y(test(c,i)),nHat))); % poisspdf or rHat=exp(X*B)
    
    % compute PVAF for test
    SS_res=var(Y(test(c,i))-nHat);
    SS_tot=var(Y(test(c,i)));
    PVAF=1-SS_res./SS_tot;
    
    % plot predicted vs actual   %&%
    if ~isempty(varargin)
%     if PVAF<-0.05 
        plot(Y(test(c,i)),nHat,'.','color',cmap(id_vis_occ,:),'markerfacecolor','w','markersize',4);
        
        disp([Y(test(c,i)) nHat]);
        
%         waitforbuttonpress; clf;
    end
end

function R=reshape_firing_rate(r,mask)
% r: [neuron x condition x time]
R=reshape(shiftdim(r,1),[],size(r,1))'; % [1385neurons × (79×100) double]
R=R(:,mask==1); % [1385neurons × k double]

function [V,O]=generate_design_matrix(vis,occ,fname_vis,fname_occ)

% V, O: [variables x condition x time] design matrix
% input:
%   vis, occ: structures
%   fname_vis, fname_occ: fieldnames

nvar=length(fname_vis);
[nrow,ncol]=size(vis.(fname_vis{1}));

V=nan(nvar,nrow,ncol);

for i=1:nvar
    V(i,:,:)=vis.(fname_vis{i});
    O(i,:,:)=occ.(fname_occ{i});
end

%%
function [vis,occ]=create_variables(vis,occ)

vis.ball_pos_dx=diff_same_size(vis.ball_pos_x);
vis.ball_pos_dy=diff_same_size(vis.ball_pos_y);

vis.ball_pos_dx_TRUE=diff_same_size(vis.ball_pos_x_TRUE);
vis.ball_pos_dy_TRUE=diff_same_size(vis.ball_pos_y_TRUE);

vis.ball_pos_dtheta=atan2(vis.ball_pos_dy,vis.ball_pos_dx);
vis.ball_pos_dspeed=sqrt((vis.ball_pos_dy.^2)+(vis.ball_pos_dx.^2));

vis.ball_pos_dtheta_TRUE=atan2(vis.ball_pos_dy_TRUE,vis.ball_pos_dx_TRUE);
vis.ball_pos_dspeed_TRUE=sqrt((vis.ball_pos_dy_TRUE.^2)+(vis.ball_pos_dx_TRUE.^2));

%% 
function y=diff_same_size(x)
% this is also how Rishi code for difference values
if numel(x)==length(x) % 1D
    y=nan(size(x));
    y(1:(length(x)-1))=diff(x);
    y(end)=y(end-1);    
else % 2D: assume time is columnwise
    y=nan(size(x));
    y(:,1:(size(x,2)-1))=diff(x,1,2);
    y(end,:)=y(end-1,:);
end
return;


%% scratch pad
return;

%% check correlation b/t independent variables
% some are highly correlated (e.g., paddle_y and joystick)
% because just no change during "off-period"
[cv,pv]=corrcoef(V');
[h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh(pv,0.05,'dep','no');
figure; imagesc(corrcoef(V')); colorbar; ha;
n_sig=0;
for i=1:length(fname_vis)
    for j=1:length(fname_vis)
        if i<j & pv(i,j)<crit_p & cv(i,j)>0.4
            disp([num2str(i) ' : ' num2str(j) '  ,  ' fname_vis{i} ' : ' fname_vis{j}]);
            plot(j,i,'k*','markersize',12);
            n_sig=n_sig+1;
        end
    end
end
n_sig
[cv,pv]=corrcoef(O');
[h, crit_p, adj_ci_cvrg, adj_p]=fdr_bh(pv,0.05,'dep','no');
figure; imagesc(corrcoef(O')); colorbar; ha;
n_sig=0;
for i=1:length(fname_occ)
    for j=1:length(fname_occ)
        if i<j & pv(i,j)<crit_p  & cv(i,j)>0.4
            disp([num2str(i) ' : ' num2str(j) '  ,  ' fname_occ{i} ' : ' fname_occ{j}]);
            plot(j,i,'k*','markersize',12);
            n_sig=n_sig+1;
        end
    end
end
n_sig

figure; imagesc(squeeze(vis.paddle_pos_y));
figure; imagesc(squeeze(vis.joy));
figure; plot(vis.paddle_pos_y(:),vis.joy(:),'o');
tmp=diff_same_size(vis.paddle_pos_y);
figure; plot(tmp(:),vis.joy(:),'o');

% conclusion: smoothing makes match worse; just use d?_TRUE
x=smoothdata(occ.ball_pos_x_TRUE,2,'gaussian');
y=smoothdata(occ.ball_pos_y_TRUE,2,'gaussian');
theta_smooth=atan2(diff_same_size(y),diff_same_size(x));
speed_smooth=sqrt((diff_same_size(y).^2)+(diff_same_size(x).^2));
figure; histogram(theta_smooth(:)-occ.ball_pos_dtheta_TRUE(:),50,'DisplayStyle','stairs');hold all;
x0=occ.ball_pos_x_TRUE;y0=occ.ball_pos_y_TRUE; % no smoothing
theta_smooth0=atan2(occ.ball_pos_dy_TRUE,occ.ball_pos_dx_TRUE);
% theta_smooth0=atan2(diff_same_size(y0),diff_same_size(x0));
histogram(theta_smooth0(:)-occ.ball_pos_dtheta_TRUE(:),50,'DisplayStyle','stairs');
speed_smooth0=sqrt((occ.ball_pos_dy_TRUE.^2)+(occ.ball_pos_dx_TRUE.^2));
% speed_smooth0=sqrt((diff_same_size(y0).^2)+(diff_same_size(x0).^2));
figure; histogram(speed_smooth(:)-occ.ball_pos_dspeed_TRUE(:),50,'DisplayStyle','stairs');hold all;
histogram(speed_smooth0(:)-occ.ball_pos_dspeed_TRUE(:),50,'DisplayStyle','stairs');hold all;
figure; plot(theta_smooth(:),occ.ball_pos_dtheta_TRUE(:),'.');
figure; plot(theta_smooth0(:),occ.ball_pos_dtheta_TRUE(:),'.');
figure; plot(speed_smooth(:),occ.ball_pos_dspeed_TRUE(:),'.');
figure; plot(speed_smooth0(:),occ.ball_pos_dspeed_TRUE(:),'.');
figure;
for i=1:size(theta_smooth,1)
    subplot(2,1,1); plot(theta_smooth(i,:));
    ha; plot(occ.ball_pos_dtheta_TRUE(i,:),'r');
    subplot(2,1,2); plot(speed_smooth(i,:));
    ha; plot(occ.ball_pos_dtheta(i,:),'r');
end

% conclusion: eye_dv/dh is consistent with dtheta; but with smoothing
i=20;
dx=(occ.ball_pos_dx_TRUE); 
dy=(occ.ball_pos_dy_TRUE);
ball_pos_theta_TRUE=atan2(dy,dx);
figure; plot(ball_pos_theta_TRUE(:),occ.ball_pos_dtheta(:),'.'); % plotCmap(ball_pos_theta_TRUE(:),occ.ball_pos_dtheta(:),0);
ball_pos_speed_TRUE=sqrt((dy.^2)+(dx.^2));
figure; plot(ball_pos_speed_TRUE(:),occ.ball_pos_dspeed(:),'.');% plotCmap(ball_pos_speed_TRUE(:),occ.ball_pos_dspeed(:),0);
 figure;plot(ball_pos_theta_TRUE(i,:));hold all;
 plot(occ.ball_pos_dtheta(i,:),'r');

 % eye_dv/dh is consistent with dtheta; but with smoothing
i=20;
 tmp=atan2(vis.eye_dv,vis.eye_dh); % theta
 figure;subplot(2,1,1); ha;  plot(tmp(i,:)); plot(vis.eye_dtheta(i,:),'r');
 tmp=sqrt((vis.eye_dv.^2)+(vis.eye_dh.^2)); % speed
subplot(2,1,2); ha; plot(tmp(i,:));plot(vis.eye_dspeed(i,:),'r');
 
% figure out how dtheta dspeed are computed
% conclusion: eye_dv/dh is consistent with dtheta; but with smoothing
i=30; t=50;
x=occ.ball_pos_x_TRUE(i,:);
y=occ.ball_pos_y_TRUE(i,:);
figure; ha;
plot(x,y,'k.');
plot([0 occ.ball_pos_dx_TRUE(i,t)]+x(t),...
    [0 occ.ball_pos_dy_TRUE(i,t)]+y(t),'r-');

 % TRUE is different
 i=30; figure; plot(D.behavioral_responses.vis.ball_pos_x(i,:),D.behavioral_responses.vis.ball_pos_y(i,:));
ha; plot(D.behavioral_responses.vis.ball_pos_x_TRUE(i,:),D.behavioral_responses.vis.ball_pos_y_TRUE(i,:),'r');

% i=30; x=diff(occ.ball_pos_x_TRUE(i,:));
% y=occ.ball_pos_dx_TRUE(i,:);
