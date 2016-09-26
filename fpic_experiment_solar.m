addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
parallel=1; subset=1; 
if subset
    fprintf('Using subset of training init ');
else
    fprintf('Using Kmeans init ');
end
warning('off','all');
%%% load and set up data %%%
load solar.mat
x=X; Y=y;
x_mean=mean(x); x_std=std(x);
y_mean=mean(y); y_std=std(y);
x = (x-x_mean)/x_std; %normalise x;
y = (y-y_mean)/y_std; %normalise y;
[n,D]=size(x);

%%%%%%%%%%%%%%%%%%%
ind=1;
if parallel
    numiter=10;
else numiter=1;
end
if parallel
    num_workers=10;
    POOL=parpool('local',num_workers);
end
m_values=[10,20,40,80,160,320];
lm=length(m_values);
lb_table=zeros(numiter,lm);
ub_fic_table = zeros(numiter,lm);
ub_pic_table = zeros(numiter,lm);

opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
for m=m_values
fprintf('m=%d \n',m);
% indices for PIC
num_blocks=ceil(n/m);
pind=cell(1,num_blocks);
for ip=1:num_blocks
    pind{ip} = (m*(ip-1)+1):min(m*ip,n);
end

parfor i=1:numiter
rng(i);
warning('off','all');
%t=getCurrentTask(); k=t.ID;
%filename=['experiment_results/mauna10m',num2str(k),'.txt'];
%fileID=fopen(filename,'at');

%%% Initialise gp_var %%%

lik=lik_init(y);
gpcf_se1=se_init(x,y);
gpcf_se2=se_init(x,y);
gpcf_se=gpcf_sum('cf',{gpcf_se1,gpcf_se2});
gpcf_per = per_init(x,y);
gpcf=gpcf_prod('cf',{gpcf_se,gpcf_per});
if subset
    [X_u,idx]=datasample(x,m,1,'Replace',false); %random initialisation
    idx_u_cell{i}=idx; %store indices of subset
else
    [~,X_u] = kmeans(x,m); %inducing pts initialised by Kmeans++
end
gp_var_loc = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', X_u);
gp_var_loc = gp_set(gp_var_loc, 'infer_params', 'covariance+likelihood');
gp_fic_loc = gp_set('type','FIC', 'lik', lik, 'cf',gpcf,'X_u', X_u);
gp_fic_loc = gp_set(gp_fic_loc, 'infer_params', 'covariance+likelihood');
gp_pic_loc = gp_set('type', 'PIC', 'lik', lik, 'cf',gpcf,'X_u', X_u,'tr_index',pind);
gp_pic_loc = gp_set(gp_pic_loc, 'infer_params', 'covariance+likelihood');

%%% Optimise gp_var/fic/pic, record ml and gp_var%%%
gp_var_loc=gp_optim(gp_var_loc,x,y,'opt',opt,'optimf',@fminscg);
[~,nll]=gp_e([],gp_var_loc,x,y);
lb_table(i,ind)=-nll;

gp_fic_loc=gp_optim(gp_fic_loc,x,y,'opt',opt,'optimf',@fminscg);
[~,nll]=gp_e([],gp_fic_loc,x,y);
ub_fic_table(i,ind)=-nll;

gp_pic_loc=gp_optim(gp_pic_loc,x,y,'opt',opt,'optimf',@fminscg);
[~,nll]=gp_e([],gp_pic_loc,x,y);
ub_pic_table(i,ind)=-nll;

fprintf('optim for worker %d done \n',i);
end

ind=ind+1;
end
delete(POOL)

%%% Optimise FIC/PIC to get approximate UBs

% just to have container for gp_fic,gp_pic gpcf %
% lik=lik_init(y);
% gpcf_se1=se_init(x,y);
% gpcf_se2=se_init(x,y);
% gpcf_se=gpcf_sum('cf',{gpcf_se1,gpcf_se2});
% gpcf_per = per_init(x,y);
% gpcf=gpcf_prod('cf',{gpcf_se,gpcf_per});
% %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% 
% gp_fic = gp_set('type', 'FIC', 'lik', lik, 'cf',gpcf,'X_u', xu);
% gp_fic = gp_set(gp_fic, 'infer_params', 'covariance+likelihood');
% num_blocks=ceil(n/m);
% pind=cell(1,num_blocks);
% for i=1:num_blocks
%     pind{i} = (m*(i-1)+1):min(m*i,n);
% end
% gp_pic = gp_set('type', 'PIC', 'lik', lik, 'cf',gpcf,'X_u', xu, 'tr_index',pind);
% gp_pic = gp_set(gp_pic, 'infer_params', 'covariance+likelihood');
% gp_fic = gp_unpak(gp_fic,w);
% gp_pic = gp_unpak(gp_pic,w); %initialise fic and pic with best VAR params
% gp_fic=gp_optim(gp_fic,x,y,'opt',opt,'optimf',@fminscg);
% gp_pic=gp_optim(gp_pic,x,y,'opt',opt,'optimf',@fminscg);
% [~,nll]=gp_e([],gp_fic,x,y);
% ub_fic_table(ind) = -nll;
% [~,nll]=gp_e([],gp_pic,x,y);
% ub_pic_table(ind) = -nll;



