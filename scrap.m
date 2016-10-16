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
solar = 0;
concrete = 1;
mauna = 0;

subset = 1;
m_values=[10,20,40,80,160,320];
%m_values=[320];
numiter=1;
lb_table=zeros(numiter,length(m_values));
approx_ub_table=zeros(numiter,length(m_values));
ind =1;
if solar
    load solar.mat
    x=X; Y=y;
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-x_mean)/x_std; %normalise x;
    y = (y-y_mean)/y_std; %normalise y;
    [n,D]=size(x);
end
if concrete
    load concrete.mat
    x=X; Y=y;
    [n,D]=size(x);
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
    y = (y-y_mean)/y_std;
end

if mauna
    load mauna.txt
    z = mauna(:,2) ~= -99.99; % get rid of missing data
    x = mauna(z,1); y = mauna(z,2); % extract year and CO2 concentration
    X = x; Y = y;
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-x_mean)/x_std; %normalise x;
    y = (y-y_mean)/y_std; %normalise y;
    [n,D]=size(x);
end

num_workers=20;
%POOL=parpool('local',num_workers);
for dim = 1:D
    fprintf('dim=%d \n',dim);
for m=m_values
    fprintf('m=%d \n',m);
for i=1:numiter
    %rng(i)
if solar
    %%% Initialise gp_var %%%
    lik=lik_init(y);
    gpcf_se1=se_init(x,y);
    gpcf_se2=se_init(x,y);
    gpcf_se=gpcf_sum('cf',{gpcf_se1,gpcf_se2});
    gpcf_per = per_init(x,y);
    gpcf=gpcf_prod('cf',{gpcf_se,gpcf_per});
    if subset
        X_u = datasample(x,m,1,'Replace',false); 
    else
        [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
    end
    gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', X_u);
end
if concrete
    %gpcf_wn=gpcf_prod('cf',{gpcf_constant(),gpcf_cat()});
    lik = lik_init(y);
    gpcf_lin=lin_init(dim);
    fprintf('lin_sigma2 = %4.3f, signal_var = %4.3f \n',gpcf_lin.coeffSigma2,lik.sigma2)
%     gpcf_se1=se_init(x(:,1),y,1);
%     gpcf_se2=se_init(x(:,2),y,2);
%     gpcf_se4=se_init(x(:,4),y,4);
%     gpcf_se7=se_init(x(:,7),y,7);
%     gpcf_se8=se_init(x(:,8),y,8);
%     gpcf1=gpcf_prod('cf',{gpcf_wn,gpcf_lin4});
%     gpcf2=gpcf_prod('cf',{gpcf_se1,gpcf_se7});
%     gpcf3=gpcf_prod('cf',{gpcf_se1,gpcf_se2,gpcf_se4});
%     gpcf4=gpcf_prod('cf',{gpcf_se2,gpcf_se4,gpcf_se8});
%     gpcf5=gpcf_prod('cf',{gpcf_se2,gpcf_se4,gpcf_se7,gpcf_se8,gpcf_lin4});
    if subset
        X_u = datasample(x,m,1,'Replace',false); 
    else
        [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
    end
    % gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3,gpcf4,gpcf5},'X_u', X_u);
    gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf_lin,'X_u', X_u);
end
if mauna
    lik=lik_init(y);
    gpcf_se1=se_init(x,y);
    gpcf_lin=lin_init();
    gpcf_se2=se_init(x,y);
    gpcf_per=per_init(x,y);
    gpcf1=gpcf_prod('cf',{gpcf_se1,gpcf_lin});
    gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
    gpcf3=se_init(x,y);
    if subset
        X_u=datasample(x,m,1,'Replace',false); %random initialisation
    else
        [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
    end
    gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3},'X_u', X_u); 
end
% gp_fic = gp_set('type', 'FIC', 'lik', lik, 'cf',gpcf,'X_u', X_u);
% num_blocks=ceil(n/m);
% myind=cell(1,num_blocks);
% for block=1:num_blocks
%     myind{block} = (m*(block-1)+1):min(m*block,n);
% end
%gp_pic = gp_set('type', 'PIC', 'lik', lik, 'cf',gpcf,'X_u', X_u, 'tr_index',ind);
%gp_dtc = gp_set('type', 'DTC', 'lik', lik, 'cf',gpcf,'X_u', X_u);
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
%gp_fic = gp_set(gp_fic, 'infer_params', 'covariance+likelihood');
%gp_pic = gp_set(gp_pic, 'infer_params', 'covariance+likelihood');
%gp_dtc = gp_set(gp_dtc, 'infer_params', 'covariance+likelihood');
opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
warning('off','all');
gp_var_new=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
%w=gp_pak(gp_var);
%gp_fic=gp_unpak(gp_fic,w);
%gp_pic=gp_unpak(gp_pic,w);
%gp_dtc = gp_optim(gp_dtc,x,y,'opt',opt,'optimf',@fminscg);
% [energy,~]=gp_e([],gp_var_new,x,y);
% lb_table(i,ind) = -energy;
%gp_e([],gp_var,x,y)
%gp_e([],gp_fic,x,y)
%gp_e([],gp_pic,x,y)
%approx_ub_grad(w,gp_var,x,y)
%gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3},'X_u', X_u); 

% if m==320
%     [gp_var,val]=approx_ub(gp_var,x,y,opt);
%     signal_var=gp_var.lik.sigma2;
%     K_mn=gp_cov(gp_var,X_u,x); K_mm=gp_trcov(gp_var,X_u);
%     L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
%     L=L_mm'\K_mn; %L'*L=K_hat=K_mn'*(K_mm\K_mn)
%     A=L*L'+signal_var*eye(m);
%     L_naive=chol(A);
%     K_naive=L'*L;
%     n_cond=cond(K_naive + signal_var*eye(n));
%     fprintf('-val =%4.3f, sigma2=%4.8f, cond=%4.3f \n',...
%         -val, gp_var.lik.sigma2,n_cond);
%     approx_ub_table(i,ind)=-val;
% else
    %[gp_var_new,val_new]=approx_ub(gp_var_new,x,y,opt);
    %[gp_var,val]=approx_ub(gp_var,x,y,opt);
    %approx_ub_table(i,ind)=-min(val,val_new);
% end
%[gp_var_new,val]=minimax(gp_var,x,y,opt);
%w=gp_pak(gp_dtc);
%gp_pic=gp_unpak(gp_pic,w);
%gp_e([],gp_pic,x,y)
%fprintf('optim for worker %d done \n',i);
end
ind = ind + 1;
end
end
%ub = -temp;
%fprintf('lb = %4.3f, ub=%4.3f \n',lb,ub);