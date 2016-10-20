% script to time how long it takes to learn ind pts
addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
m_values=[10,20,40,80,160,320];
numiter=10;

num_workers=10;
POOL=parpool('local',num_workers);

solar=0;
mauna=0;
concrete=1;

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

times = zeros(numiter,length(m_values));
for ind =1:length(m_values)
    m = m_values(ind);
    fprintf('m=%d \n',m);
    parfor iter = 1:numiter
        warning('off','all');
        if solar
            lik=lik_init(y);
            gpcf_se1=se_init(x,y);
            gpcf_se2=se_init(x,y);
            gpcf_se=gpcf_sum('cf',{gpcf_se1,gpcf_se2});
            gpcf_per = per_init(x,y);
            gpcf=gpcf_prod('cf',{gpcf_se,gpcf_per});
            X_u = datasample(x,m,1,'Replace',false);
            gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', X_u);
        end
        if concrete
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
            [X_u,idx]=datasample(x,m,1,'Replace',false); %random initialisation
            gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3,gpcf4,gpcf5},'X_u', X_u);
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
            [X_u,idx]=datasample(x,m,1,'Replace',false); %random initialisation
            gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3},'X_u', X_u);

        end
        gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
        opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
        tic; 
        gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
        time = toc;
        times(iter,ind)=time;
    end
end
mean(times)
std(times)