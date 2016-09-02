addpath(genpath('/homes/hkim/Documents/GPstuff-4.6')); 
concrete = 0;
solar = 1;
mauna = 0;

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
    gpcf_se=se_init(x,y);
    gpcf_per = per_init(x,y);
    gpcf=gpcf_prod('cf',{gpcf_se,gpcf_per});
    gp=gp_set('lik',lik,'cf',gpcf);
    gp=gp_optim(gp,x,y,'opt',opt);
    pred=gp_pred(gp,x,y,x);
    [~,nll]=gp_e([],gp,x,y);
    pred=pred*y_std+y_mean;
    figure();
    scatter(X,Y);
    hold on
    plot(X,pred);
    fprintf('ml = %4.3f \n',-nll);
    fprintf('PER magnSigma2=%4.3f, period=%4.3f \n',...
        gp.cf{1}.cf{2}.magnSigma2, gp.cf{1}.cf{2}.period);
    fprintf('SE magnSigma2=%4.3f, l=%4.3f \n',...
        gp.cf{1}.cf{1}.magnSigma2, gp.cf{1}.cf{1}.lengthScale);
    m_values=[10,20,40,80,160,320];
    if 1==0
    nll=zeros(1,length(m_values));
    RMSE=zeros(1,length(m_values));
    warning('off','all');
    figure();
    for i=1:length(m_values)
        rng(i);
        m=m_values(i);
        fprintf('m=%d \n',m);
        [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
        gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', X_u); 
        gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
        opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
        gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
        [~,nll(i)]=gp_e([],gp_var,x,y);
        pred=gp_pred(gp_var,x,y,x);
        pred=pred*y_std+y_mean;
        RMSE(i)=sqrt(sum((Y-pred).^2)/n);
        subplot(length(m_values),1,i)
        scatter(X,Y);
        hold on
        plot(X,pred);
        %scatter(X_u,1362*ones(m,1))
        %scatter(gp_var.X_u,1360*ones(m,1))
        hold off
        fprintf('PER magnSigma2=%4.3f, period=%4.3f \n',...
            gp_var.cf{1}.cf{2}.magnSigma2, gp_var.cf{1}.cf{2}.period);
        fprintf('SE magnSigma2=%4.3f, l=%4.3f \n',...
            gp_var.cf{1}.cf{1}.magnSigma2, gp_var.cf{1}.cf{1}.lengthScale);
    end
    figure(1000); plot(-nll);
    figure(1001); plot(RMSE);
    end
end

if mauna
    load mauna.txt
    z = mauna(:,2) ~= -99.99; % get rid of missing data
    x = mauna(z,1); y = mauna(z,2); % extract year and CO2 concentration
    x_mean=mean(x); x_std=std(x);
    y_mean=mean(y); y_std=std(y);
    xw = (x-x_mean)/x_std; %normalise x;
    yw = (y-y_mean)/y_std; %normalise y;
    [n,D]=size(x);
    
    l1=0.10;sf1=0.1;cs2=1.00;l2=1.00;sf2=0.1;lper=1.00;sfper=0.1;l3=1.00;sf3=0.1;
    per=1/x_std; signal_var=0.1;
    %pl=prior_gaussian('s2',0.5);
    pl=prior_t();
    lik = lik_gaussian('sigma2', signal_var);
    gpcf_se1 = gpcf_sexp('lengthScale', l1, 'magnSigma2',sf1,'lengthScale_prior',pl); 
    gpcf_lin=gpcf_linear('coeffSigma2',cs2);
    gpcf1=gpcf_prod('cf',{gpcf_se1,gpcf_lin});
    gpcf_se2 = gpcf_sexp('lengthScale', l2, 'magnSigma2', sf2,'lengthScale_prior',pl);
    gpcf_per = gpcf_periodic('lengthScale',lper,'period',per,'magnSigma2',sfper,'lengthScale_prior',pl);
    gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
    gpcf3 = gpcf_sexp('lengthScale', l3, 'magnSigma2', sf3,'lengthScale_prior',pl); 

    %gp=gp_set('lik',lik,'cf',{gpcf1,gpcf2,gpcf3});
    %opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','iter','MaxIter',1000);
    %gp=gp_optim(gp,xw,yw,'opt',opt);
    m_values=[10,20,40,80];
    nll=zeros(1,length(m_values));
    RMSE=zeros(1,length(m_values));
    warning('off','all');
    
    for i=1:length(m_values)
        rng(i);
        m=m_values(i);
        fprintf('m=%d \n',m);
        [~,X_u]=kmeans(xw,m); %inducing pts initialised by Kmeans++
        gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3},'X_u', X_u); 
        gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
        opt=optimset('TolFun',1e-2,'TolX',1e-3,'Display','iter','MaxIter',1000);
        gp_var=gp_optim(gp_var,xw,yw,'opt',opt,'optimf',@fminscg);
        [~,nll(i)]=gp_e([],gp_var,xw,yw);
        pred=gp_pred(gp_var,xw,yw,xw);
        pred=pred*y_std+y_mean;
        RMSE(i)=sqrt(sum((y-pred).^2)/n);
        
        %subplot(length(m_values),1,i)
        figure();
        scatter(x,y);
        hold on
        plot(x,pred);
        %scatter(X_u,450*ones(m,1))
        %scatter(gp_var.X_u,300*ones(m,1))
        hold off
    end

end
