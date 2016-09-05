addpath(genpath('/homes/hkim/Documents/GPstuff-4.6')); 
addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
concrete = 0;
solar = 0;
mauna = 1;
warning('off','all');
POOL=parpool('local',4);
if concrete
load concrete.mat
num_workers=10;
%POOL=parpool('local',num_workers);

x=X;
[n,D]=size(x);
x_mean=mean(x); x_std=std(x);
y_mean=mean(y); y_std=std(y);
xw = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
yw = (y-y_mean)/y_std; %normalise y;

pl=prior_gaussian('s2',0.5);
lik = lik_gaussian();
gpcf_wn=gpcf_prod('cf',{gpcf_constant(),gpcf_cat()});
gpcf= gpcf_sexp('lengthScale_prior',pl); 
gpcf_lin4=gpcf_linear('selectedVariables',4);
gpcf_se1=gpcf_sexp('selectedVariable',1,'lengthScale_prior',pl);
gpcf_se2=gpcf_sexp('selectedVariable',2,'lengthScale_prior',pl);
gpcf_se4=gpcf_sexp('selectedVariable',4,'lengthScale_prior',pl);
gpcf_se7=gpcf_sexp('selectedVariable',7,'lengthScale_prior',pl);
gpcf_se8=gpcf_sexp('selectedVariable',8,'lengthScale_prior',pl);
gpcf1=gpcf_prod('cf',{gpcf_wn,gpcf_lin4});
gpcf2=gpcf_prod('cf',{gpcf_se1,gpcf_se7});
gpcf3=gpcf_prod('cf',{gpcf_se1,gpcf_se2,gpcf_se4});
gpcf4=gpcf_prod('cf',{gpcf_se2,gpcf_se4,gpcf_se8});
gpcf5=gpcf_prod('cf',{gpcf_se2,gpcf_se4,gpcf_se7,gpcf_se8,gpcf_lin4});

gp=gp_set('lik',lik,'cf',{gpcf1,gpcf2,gpcf3,gpcf4,gpcf5});
opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','iter','MaxIter',1000);
gp=gp_optim(gp,xw,yw,'opt',opt);
[~,nll]=gp_e([],gp,xw,yw);

pred=gp_pred(gp,xw,yw,xw);
pred=pred*y_std+y_mean;
RMSE=sqrt(sum((y-pred).^2)/n);

%delete(POOL)
end

if solar
    load solar.mat
    x=X; Y=y;
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    x = (x-x_mean)/x_std; %normalise x;
    y = (y-y_mean)/y_std; %normalise y;
    [n,D]=size(x);
    
    lik=lik_init(y);
    gpcf_se1=se_init(x,y);
    gpcf_se2=se_init(x,y);
    gpcf_se=gpcf_sum('cf',{gpcf_se1,gpcf_se2});
    gpcf_per = per_init(x,y);
    gpcf=gpcf_prod('cf',{gpcf_se,gpcf_per});
    opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
    gp=gp_set('lik',lik,'cf',gpcf);
    gp=gp_optim(gp,x,y,'opt',opt);
    pred=gp_pred(gp,x,y,x);
    [~,nll]=gp_e([],gp,x,y);
    pred=pred*y_std+y_mean;
    figure();
    scatter(X,Y,'x');
    hold on
    plot(X,pred);
    fprintf('ml = %4.3f \n',-nll);
    fprintf('PER magnSigma2=%4.3f, period=%4.3f, l=%4.3f \n',...
        gp.cf{1}.cf{2}.magnSigma2, gp.cf{1}.cf{2}.period, gp.cf{1}.cf{2}.lengthScale);
    fprintf('SE1 magnSigma2=%4.3f, l=%4.3f \n',...
        gp.cf{1}.cf{1}.cf{1}.magnSigma2, gp.cf{1}.cf{1}.cf{1}.lengthScale);
    fprintf('SE2 magnSigma2=%4.3f, l=%4.3f \n',...
        gp.cf{1}.cf{1}.cf{2}.magnSigma2, gp.cf{1}.cf{1}.cf{2}.lengthScale);

    m_values=[10,20,40,80,160,320];
    if 1==1
    nll=zeros(1,length(m_values));
    RMSE=zeros(1,length(m_values));
    figure();
    for i=1:length(m_values)
        m=m_values(i);
        fprintf('m=%d \n',m);
        [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
        gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', X_u); 
        gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
        gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
        [~,nll(i)]=gp_e([],gp_var,x,y);
        pred=gp_pred(gp_var,x,y,x);
        pred=pred*y_std+y_mean;
        RMSE(i)=sqrt(sum((Y-pred).^2)/n);
        subplot(length(m_values),1,i)
        scatter(X,Y,'x');
        hold on
        plot(X,pred);
        %scatter(X_u,1362*ones(m,1))
        %scatter(gp_var.X_u,1360*ones(m,1))
        hold off
        fprintf('PER magnSigma2=%4.3f, period=%4.3f \n',...
            gp_var.cf{1}.cf{2}.magnSigma2, gp_var.cf{1}.cf{2}.period);
        fprintf('SE1 magnSigma2=%4.3f, l=%4.3f \n',...
            gp_var.cf{1}.cf{1}.cf{1}.magnSigma2, gp_var.cf{1}.cf{1}.cf{1}.lengthScale);
        fprintf('SE2 magnSigma2=%4.3f, l=%4.3f \n',...
            gp_var.cf{1}.cf{1}.cf{2}.magnSigma2, gp_var.cf{1}.cf{1}.cf{2}.lengthScale);
    end
    figure(1000); plot(-nll);
    figure(1001); plot(RMSE);
    end
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
    
    l1=0.10;sf1=0.1;cs2=1.00;l2=1.00;sf2=0.1;lper=1.00;sfper=0.1;l3=1.00;sf3=0.1;
    per=1/x_std; signal_var=0.1;
    %pl=prior_gaussian('s2',0.5);
%     pl=prior_t();
%     lik = lik_gaussian('sigma2', signal_var);
%     gpcf_se1 = gpcf_sexp('lengthScale', l1, 'magnSigma2',sf1,'lengthScale_prior',pl); 
%     gpcf_lin=gpcf_linear('coeffSigma2',cs2);
%     gpcf_se2 = gpcf_sexp('lengthScale', l2, 'magnSigma2', sf2,'lengthScale_prior',pl);
%     gpcf_per = gpcf_periodic('lengthScale',lper,'period',per,'magnSigma2',sfper,'lengthScale_prior',pl);
%     gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
%     gpcf3 = gpcf_sexp('lengthScale', l3, 'magnSigma2', sf3,'lengthScale_prior',pl);
    parfor i=1:10
    warning('off','all');
    lik=lik_init(y);
    gpcf_se1=se_init(x,y);
    gpcf_lin=lin_init();
    gpcf_se2=se_init(x,y);
    gpcf_per=per_init(x,y);
    gpcf1=gpcf_prod('cf',{gpcf_se1,gpcf_lin});
    gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
    gpcf3=se_init(x,y);
    opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
    gp=gp_set('lik',lik,'cf',{gpcf1,gpcf2,gpcf3});
    gp=gp_optim(gp,x,y,'opt',opt);
    pred=gp_pred(gp,x,y,x);
    [~,nll]=gp_e([],gp,x,y);
    pred=pred*y_std+y_mean;
    figure();
    scatter(X,Y,'x');
    hold on
    plot(X,pred);
    hold off
    fprintf('ml = %4.3f \n',-nll);
    fprintf('SE1 magnSigma2=%4.3f, l=%4.3f \n',...
        gp.cf{1}.cf{1}.magnSigma2, gp.cf{1}.cf{1}.lengthScale);
    fprintf('LIN coeffSigma2=%4.3f \n',...
        gp.cf{1}.cf{2}.coeffSigma2);
    fprintf('SE2 magnSigma2=%4.3f, l=%4.3f \n',...
        gp.cf{2}.cf{1}.magnSigma2, gp.cf{2}.cf{1}.lengthScale);
    fprintf('PER magnSigma2=%4.3f, l=%4.3f, per=%4.3f \n',...
        gp.cf{2}.cf{2}.magnSigma2, gp.cf{2}.cf{2}.lengthScale, gp.cf{2}.cf{2}.period);
    fprintf('SE3 magnSigma2=%4.3f, l=%4.3f \n',...
        gp.cf{3}.magnSigma2, gp.cf{3}.lengthScale);
    end
    delete(POOL)
    m_values=[10,20,40,80,160,320];
    if 1==0
    nll=zeros(1,length(m_values));
    RMSE=zeros(1,length(m_values));
    
    for i=1:length(m_values)
        m=m_values(i);
        fprintf('m=%d \n',m);
        [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
        gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3},'X_u', X_u); 
        gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
        opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
        gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
        [~,nll(i)]=gp_e([],gp_var,x,y);
        pred=gp_pred(gp_var,x,y,x);
        pred=pred*y_std+y_mean;
        RMSE(i)=sqrt(sum((Y-pred).^2)/n);
        
        subplot(length(m_values),1,i)
        scatter(X,Y,'x');
        hold on
        plot(X,pred);
        %scatter(X_u,450*ones(m,1))
        %scatter(gp_var.X_u,300*ones(m,1))
        hold off
        fprintf('SE1 magnSigma2=%4.3f, l=%4.3f \n',...
            gp_var.cf{1}.cf{1}.magnSigma2, gp_var.cf{1}.cf{1}.lengthScale);
        fprintf('LIN coeffSigma2=%4.3f \n',...
            gp_var.cf{1}.cf{2}.coeffSigma2);
        fprintf('SE2 magnSigma2=%4.3f, l=%4.3f \n',...
            gp_var.cf{2}.cf{1}.magnSigma2, gp_var.cf{2}.cf{1}.lengthScale);
        fprintf('PER magnSigma2=%4.3f, l=%4.3f, per=%4.3f \n',...
            gp_var.cf{2}.cf{2}.magnSigma2, gp_var.cf{2}.cf{2}.lengthScale, gp_var.cf{2}.cf{2}.period);
        fprintf('SE3 magnSigma2=%4.3f, l=%4.3f \n',...
            gp_var.cf{3}.magnSigma2, gp_var.cf{3}.lengthScale);
    end
    figure(1000); plot(-nll);
    figure(1001); plot(RMSE);
    end
end
