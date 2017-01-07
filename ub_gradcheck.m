addpath(genpath('/homes/hkim/SSDGP/GPstuff-4.6'));
addpath(genpath('/Users/hyunjik11/Documents/SSDGP/GPstuff-4.6'));
solar = 0;
concrete = 1;
mauna = 0;
if solar
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
    if subset
        X_u = datasample(x,m,1,'Replace',false); 
    else
        [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
    end
    gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', X_u);
end
if concrete
    load concrete.mat
    x=X; Y=y;
    [n,D]=size(x);
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
    y = (y-y_mean)/y_std;
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
        X_u = datasample(x,m,1,'Replace',false); 
    else
        [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
    end
    gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3,gpcf4,gpcf5},'X_u', X_u);
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
%%% Optimise gp_var %%%
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
% opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','iter','MaxIter',1000);
% gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
w = gp_pak(gp_var); % log params
[val,grad]=ub_grad(w,gp_var,x,y);
for i=1:length(w)
    fprintf('diff_grad = ')
    for eps = [1e-2,1e-3,1e-4,1e-5,1e-6]
        z=w; z(i)=z(i)+eps;
        [val_new,grad_new] = ub_grad(z,gp_var,x,y);
        z=w; z(i)=z(i)-eps;
        [val_old,grad_old] = ub_grad(z,gp_var,x,y);
        fd = (val_new - val_old)/(2*eps);
        fprintf('%4.8f ',abs((fd-grad(i))/grad(i)))
    end
    fprintf('\n')
end
