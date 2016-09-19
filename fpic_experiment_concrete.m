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
load concrete.mat
x=X; Y=y;
[n,D]=size(x);
x_mean=mean(x); x_std=std(x);
y_mean=mean(y); y_std=std(y);
x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
y = (y-y_mean)/y_std; %normalise y;

%%%%%%%%%%%%%%%%%%%
ind=1;
if parallel
    numiter=10;
else numiter=1;
end
if parallel
    num_workers=4;
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

gpcf_wn=gpcf_prod('cf',{gpcf_constant(),gpcf_cat()});
lik = lik_init(y);
gpcf_lin4=lin_init(4);
gpcf_se1=se_init(x(:,1),y,1);
gpcf_se2=se_init(x(:,2),y,2);
gpcf_se4=se_init(x(:,4),y,4);
gpcf_se7=se_init(x(:,7),y,7);
gpcf_se8=se_init(x(:,8),y,8);
gpcf1=gpcf_prod('cf',{gpcf_wn,gpcf_lin4});
gpcf2=gpcf_prod('cf',{gpcf_se1,gpcf_se7});
gpcf3=gpcf_prod('cf',{gpcf_se1,gpcf_se2,gpcf_se4});
gpcf4=gpcf_prod('cf',{gpcf_se2,gpcf_se4,gpcf_se8});
gpcf5=gpcf_prod('cf',{gpcf_se2,gpcf_se4,gpcf_se7,gpcf_se8,gpcf_lin4});

if subset
    [X_u,idx]=datasample(x,m,1,'Replace',false); %random initialisation
    idx_u_cell{i}=idx; %store indices of subset
else
    [~,X_u] = kmeans(x,m); %inducing pts initialised by Kmeans++
end
gp_var_loc = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3,gpcf4,gpcf5},'X_u', X_u);
gp_var_loc = gp_set(gp_var_loc, 'infer_params', 'covariance+likelihood');
gp_fic_loc = gp_set('type','FIC', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3,gpcf4,gpcf5},'X_u', X_u);
gp_fic_loc = gp_set(gp_fic_loc, 'infer_params', 'covariance+likelihood');
gp_pic_loc = gp_set('type', 'PIC', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3,gpcf4,gpcf5},'X_u', X_u,'tr_index',pind);
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