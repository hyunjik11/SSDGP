% subplot(1,3,1);
% ylim([-450,-50]);
% subplot(1,3,2);
% ylim([200,650]);
% subplot(1,3,3);
% ylim([-200,-165]);
% 
% subplot(1,3,1);
% ylim([400,2200]);
% subplot(1,3,2);
% ylim([2400,3100]);
% subplot(1,3,3);
% ylim([-400,0]);
% 
% subplot(1,3,1);
% ylim([-1200,-200]);
% subplot(1,3,2);
% ylim([650,1150]);
% subplot(1,3,3);
% ylim([-520,-400]);
addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
load solar.mat
x=X; Y=y;
x_mean=mean(x); x_std=std(x);
y_mean=mean(y); y_std=std(y);
x = (x-x_mean)/x_std; %normalise x;
y = (y-y_mean)/y_std; %normalise y;
[n,D]=size(x);
subset = 1;
m=10;
%%% Initialise gp_var %%%
lik=lik_init(y);
gpcf_se1=se_init(x,y);
gpcf_se2=se_init(x,y);
gpcf_se=gpcf_sum('cf',{gpcf_se1,gpcf_se2});
gpcf_per = per_init(x,y);
gpcf=gpcf_prod('cf',{gpcf_se,gpcf_per});

%%% Optimise gp_var %%%
if subset
    X_u = datasample(x,m,1,'Replace',false); 
else
    [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
end
gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', X_u); 
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','iter','MaxIter',1000);
warning('off','all');
gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
[temp,~]=gp_e([],gp_var,x,y);
lb = -temp;
[gp_var, temp]=minimax(gp_var,x,y,opt);
ub = -temp;
fprintf('lb = %4.3f, ub=%4.3f',lb,ub);