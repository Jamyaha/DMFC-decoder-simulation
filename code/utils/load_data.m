function data=load_data(varargin)

% load trial average data
% input: varargin (subject_id, condition, timebinsize)
% e.g., 'perle_hand_dmfc', 'occ', 50

% dependence: v2struct

%%
if ~isempty(varargin)
    subject_id=varargin{1};
    condition=varargin{2};
    timebinsize=varargin{3};
    
else % default
    subject_id='mahler_hand_dmfc';
    condition='occ';
    timebinsize=50;
    
end

%% main
path_data = uigetdir; % '/Users/hansem/Dropbox (MIT)/MPong/phys/data/';
name_data = fullfile(path_data,[subject_id '_dataset_' num2str(timebinsize) 'ms.mat']);
data=load(name_data);

data=get_id_bounce(data);
    

function D=get_id_bounce(D)
% get id_bounce for visible & occluded epoch
% id_bounce_vis
% id_bounce_occ

threshold=0.1; % to find bounce

v2struct(D.behavioral_responses); % vis occ
[nCond,nT]=size(vis.ball_pos_x);

mask_vis=D.masks.vis.start_occ_pad0; % [79 x 100]
mask_occ=D.masks.occ.occ_end_pad0;

D.id_bounce_vis=[];
D.id_bounce_occ=[];

for iCond=1:nCond
    
    y0=vis.ball_pos_y(iCond,mask_vis(iCond,:)==1);
    y1=occ.ball_pos_y(iCond,mask_occ(iCond,:)==1);
    
    id_bounce_vis=sum(abs(diff(diff(y0)))>threshold)>0;
    id_bounce_occ=sum(abs(diff(diff(y1)))>threshold)>0;
%     if id_bounce_vis
%         disp([num2str(iCond) ': bounce in visible']);
%     end
%     if id_bounce_occ
%         disp([num2str(iCond) ': bounce in occluded']);
%     end
    
    D.id_bounce_vis=[D.id_bounce_vis; id_bounce_vis];
    D.id_bounce_occ=[D.id_bounce_occ; id_bounce_occ];

end

D.behavioral_responses.vis.id_bounce_vis=repmat(D.id_bounce_vis(:),1,size(D.behavioral_responses.vis.ball_pos_x,2));
D.behavioral_responses.occ.id_bounce_occ=repmat(D.id_bounce_occ(:),1,size(D.behavioral_responses.occ.ball_pos_x,2));