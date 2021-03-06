addpath(genpath('/homes/hkim/Documents/GPstuff-4.6')); 
addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
concrete = 0;
solar = 0;
mauna = 1;

numiter=1;
warning('off','all');
num_workers=10;
%POOL=parpool('local',num_workers);

table=zeros(1,numiter);
if concrete
load concrete.mat

x=X; Y=y;
[n,D]=size(x);
x_mean=mean(x); x_std=std(x);
y_mean=mean(y); y_std=std(y);
x = (x-repmat(x_mean,n,1))./repmat(x_std,n,1); %normalise x;
y = (y-y_mean)/y_std; %normalise y;

% pl=prior_gaussian('s2',0.5);
% lik = lik_gaussian();
parfor i=1:numiter
gpcf_wn=gpcf_prod('cf',{gpcf_constant(),gpcf_cat()});
% gpcf= gpcf_sexp('lengthScale_prior',pl); 
% gpcf_lin4=gpcf_linear('selectedVariables',4);
% gpcf_se1=gpcf_sexp('selectedVariables',1,'lengthScale_prior',pl);
% gpcf_se2=gpcf_sexp('selectedVariables',2,'lengthScale_prior',pl);
% gpcf_se4=gpcf_sexp('selectedVariables',4,'lengthScale_prior',pl);
% gpcf_se7=gpcf_sexp('selectedVariables',7,'lengthScale_prior',pl);
% gpcf_se8=gpcf_sexp('selectedVariables',8,'lengthScale_prior',pl);
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

gp=gp_set('lik',lik,'cf',{gpcf1,gpcf2,gpcf3,gpcf4,gpcf5});
opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
warning('off','all');
gp=gp_optim(gp,x,y,'opt',opt);
[energy,nll]=gp_e([],gp,x,y);
table(i)=-energy;
% pred=gp_pred(gp,x,y,x);
% pred=pred*y_std+y_mean;
% RMSE=sqrt(sum((Y-pred).^2)/n);
% fprintf('k1: c=%4.3f, coeffSigma2=%4.3f \n',...
%     gp.cf{1}.cf{1}.cf{1}.constSigma2,gp.cf{1}.cf{2}.coeffSigma2);
% fprintf('k2: SE1 s2=%4.3f, l=%4.3f \n', ...
%     gp.cf{2}.cf{1}.magnSigma2,gp.cf{2}.cf{1}.lengthScale(1));
% fprintf('    SE7 s2=%4.3f, l=%4.3f \n', ...
%     gp.cf{2}.cf{2}.magnSigma2,gp.cf{2}.cf{2}.lengthScale(1));
% fprintf('k3: SE1 s2=%4.3f, l=%4.3f \n', ...
%     gp.cf{3}.cf{1}.magnSigma2,gp.cf{3}.cf{1}.lengthScale(1));
% fprintf('    SE2 s2=%4.3f, l=%4.3f \n', ...
%     gp.cf{3}.cf{2}.magnSigma2,gp.cf{3}.cf{2}.lengthScale(1));
% fprintf('    SE4 s2=%4.3f, l=%4.3f \n', ...
%     gp.cf{3}.cf{3}.magnSigma2,gp.cf{3}.cf{3}.lengthScale(1));
% fprintf('k4: SE2 s2=%4.3f, l=%4.3f \n', ...
%     gp.cf{4}.cf{1}.magnSigma2,gp.cf{4}.cf{1}.lengthScale(1));
% fprintf('    SE4 s2=%4.3f, l=%4.3f \n', ...
%     gp.cf{4}.cf{2}.magnSigma2,gp.cf{4}.cf{2}.lengthScale(1));
% fprintf('    SE8 s2=%4.3f, l=%4.3f \n', ...
%     gp.cf{4}.cf{3}.magnSigma2,gp.cf{4}.cf{3}.lengthScale(1));
% fprintf('k5: SE2 s2=%4.3f, l=%4.3f \n', ...
%     gp.cf{5}.cf{1}.magnSigma2,gp.cf{5}.cf{1}.lengthScale(1));
% fprintf('    SE4 s2=%4.3f, l=%4.3f \n', ...
%     gp.cf{5}.cf{2}.magnSigma2,gp.cf{5}.cf{2}.lengthScale(1));
% fprintf('    SE7 s2=%4.3f, l=%4.3f \n', ...
%     gp.cf{5}.cf{3}.magnSigma2,gp.cf{5}.cf{3}.lengthScale(1));
% fprintf('    SE8 s2=%4.3f, l=%4.3f \n', ...
%     gp.cf{5}.cf{4}.magnSigma2,gp.cf{5}.cf{4}.lengthScale(1));
% fprintf('lik sigma2=%4.8f \n',gp.lik.sigma2);
%     m_values=[10,20,40,80,160,320];
%     if 1==0
%     nll=zeros(1,length(m_values));
%     RMSE=zeros(1,length(m_values));
%     %figure();
%     parfor i=1:length(m_values)
%         warning('off','all');
%         m=m_values(i);
%         %X_u = datasample(x,m,1,'Replace',false); 
%         [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
%         gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3,gpcf4,gpcf5},'X_u', X_u); 
%         gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
%         opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
%         gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
%         [~,nll(i)]=gp_e([],gp_var,x,y);
%         pred=gp_pred(gp_var,x,y,x);
%         pred=pred*y_std+y_mean;
%         RMSE(i)=sqrt(sum((Y-pred).^2)/n);
%         fprintf('m=%d, ml=%4.3f \n',m,-nll(i));
%     end
%     figure(1000); plot(-nll);
%     figure(1001); plot(RMSE);
%     end
end
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
    
    parfor i=1:numiter
    lik=lik_init(y);
    gpcf_se1=se_init(x,y);
    gpcf_se2=se_init(x,y);
    gpcf_se=gpcf_sum('cf',{gpcf_se1,gpcf_se2});
    gpcf_per = per_init(x,y);
    gpcf=gpcf_prod('cf',{gpcf_se,gpcf_per});
    opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','iter','MaxIter',1000);
    gp=gp_set('lik',lik,'cf',gpcf);
    warning('off','all');
    gp=gp_optim(gp,x,y,'opt',opt);
    %pred=gp_pred(gp,x,y,x);
    [energy,nll]=gp_e([],gp,x,y);
    table(i)=-energy
%     pred=pred*y_std+y_mean;
%     figure();
%     scatter(X,Y,'x');
%     hold on
%     plot(X,pred);
%     fprintf('ml = %4.3f \n',-nll);
%     fprintf('PER magnSigma2=%4.3f, period=%4.3f, l=%4.3f \n',...
%         gp.cf{1}.cf{2}.magnSigma2, gp.cf{1}.cf{2}.period, gp.cf{1}.cf{2}.lengthScale);
%     fprintf('SE1 magnSigma2=%4.3f, l=%4.3f \n',...
%         gp.cf{1}.cf{1}.cf{1}.magnSigma2, gp.cf{1}.cf{1}.cf{1}.lengthScale);
%     fprintf('SE2 magnSigma2=%4.3f, l=%4.3f \n',...
%         gp.cf{1}.cf{1}.cf{2}.magnSigma2, gp.cf{1}.cf{1}.cf{2}.lengthScale);

%     m_values=[10,20,40,80,160,320];
%     if 1==0
%     nll=zeros(1,length(m_values));
%     RMSE=zeros(1,length(m_values));
%     figure();
%     for i=1:length(m_values)
%         m=m_values(i);
%         fprintf('m=%d \n',m);
%         [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
%         gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', X_u); 
%         gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
%         gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
%         [~,nll(i)]=gp_e([],gp_var,x,y);
%         pred=gp_pred(gp_var,x,y,x);
%         pred=pred*y_std+y_mean;
%         RMSE(i)=sqrt(sum((Y-pred).^2)/n);
%         subplot(length(m_values),1,i)
%         scatter(X,Y,'x');
%         hold on
%         plot(X,pred);
%         %scatter(X_u,1362*ones(m,1))
%         %scatter(gp_var.X_u,1360*ones(m,1))
%         hold off
%         fprintf('PER magnSigma2=%4.3f, period=%4.3f \n',...
%             gp_var.cf{1}.cf{2}.magnSigma2, gp_var.cf{1}.cf{2}.period);
%         fprintf('SE1 magnSigma2=%4.3f, l=%4.3f \n',...
%             gp_var.cf{1}.cf{1}.cf{1}.magnSigma2, gp_var.cf{1}.cf{1}.cf{1}.lengthScale);
%         fprintf('SE2 magnSigma2=%4.3f, l=%4.3f \n',...
%             gp_var.cf{1}.cf{1}.cf{2}.magnSigma2, gp_var.cf{1}.cf{1}.cf{2}.lengthScale);
%     end
%     figure(1000); plot(-nll);
%     figure(1001); plot(RMSE);
%     end
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
    
%     l1=0.10;sf1=0.1;cs2=1.00;l2=1.00;sf2=0.1;lper=1.00;sfper=0.1;l3=1.00;sf3=0.1;
%     per=1/x_std; signal_var=0.1;
    %pl=prior_gaussian('s2',0.5);
%     pl=prior_t();
%     lik = lik_gaussian('sigma2', signal_var);
%     gpcf_se1 = gpcf_sexp('lengthScale', l1, 'magnSigma2',sf1,'lengthScale_prior',pl); 
%     gpcf_lin=gpcf_linear('coeffSigma2',cs2);
%     gpcf_se2 = gpcf_sexp('lengthScale', l2, 'magnSigma2', sf2,'lengthScale_prior',pl);
%     gpcf_per = gpcf_periodic('lengthScale',lper,'period',per,'magnSigma2',sfper,'lengthScale_prior',pl);
%     gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
%     gpcf3 = gpcf_sexp('lengthScale', l3, 'magnSigma2', sf3,'lengthScale_prior',pl);
    for i=1:numiter
    warning('off','all');
    lik=lik_init(y);
    gpcf_se1=se_init(x,y);
    gpcf_lin=lin_init();
    gpcf_se2=se_init(x,y);
    gpcf_per=per_init(x,y);
    gpcf1=gpcf_prod('cf',{gpcf_se1,gpcf_lin});
    gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
    gpcf3=se_init(x,y);
    opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','iter','MaxIter',1000);
    gpcf = gpcf_sum('cf',{gpcf1,gpcf2,gpcf3});
    X_u = datasample(x,m,1,'Replace',false); 
    gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', X_u);
    %gp=gp_set('lik',lik,'cf',gpcf);
    gp_var=gp_optim(gp_var,x,y,'opt',opt);
    %pred=gp_pred(gp,x,y,x);
    %[energy,nll]=gp_e([],gp,x,y);
    %table(i)=-energy;
%     pred=pred*y_std+y_mean;
%     figure();
%     scatter(X,Y,'x');
%     hold on
%     plot(X,pred);
%     hold off
%     fprintf('ml = %4.3f \n',-nll);
%     fprintf('SE1 magnSigma2=%4.3f, l=%4.3f \n',...
%         gp.cf{1}.cf{1}.magnSigma2, gp.cf{1}.cf{1}.lengthScale);
%     fprintf('LIN coeffSigma2=%4.3f \n',...
%         gp.cf{1}.cf{2}.coeffSigma2);
%     fprintf('SE2 magnSigma2=%4.3f, l=%4.3f \n',...
%         gp.cf{2}.cf{1}.magnSigma2, gp.cf{2}.cf{1}.lengthScale);
%     fprintf('PER magnSigma2=%4.3f, l=%4.3f, per=%4.3f \n',...
%         gp.cf{2}.cf{2}.magnSigma2, gp.cf{2}.cf{2}.lengthScale, gp.cf{2}.cf{2}.period);
%     fprintf('SE3 magnSigma2=%4.3f, l=%4.3f \n',...
%         gp.cf{3}.magnSigma2, gp.cf{3}.lengthScale);

    
%     m_values=[10,20,40,80,160,320];
%     if 1==0
%     nll=zeros(1,length(m_values));
%     RMSE=zeros(1,length(m_values));
%     
%     for i=1:length(m_values)
%         m=m_values(i);
%         fprintf('m=%d \n',m);
%         [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
%         gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3},'X_u', X_u); 
%         gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
%         opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
%         gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
%         [~,nll(i)]=gp_e([],gp_var,x,y);
%         pred=gp_pred(gp_var,x,y,x);
%         pred=pred*y_std+y_mean;
%         RMSE(i)=sqrt(sum((Y-pred).^2)/n);
%         
%         subplot(length(m_values),1,i)
%         scatter(X,Y,'x');
%         hold on
%         plot(X,pred);
%         %scatter(X_u,450*ones(m,1))
%         %scatter(gp_var.X_u,300*ones(m,1))
%         hold off
%         fprintf('SE1 magnSigma2=%4.3f, l=%4.3f \n',...
%             gp_var.cf{1}.cf{1}.magnSigma2, gp_var.cf{1}.cf{1}.lengthScale);
%         fprintf('LIN coeffSigma2=%4.3f \n',...
%             gp_var.cf{1}.cf{2}.coeffSigma2);
%         fprintf('SE2 magnSigma2=%4.3f, l=%4.3f \n',...
%             gp_var.cf{2}.cf{1}.magnSigma2, gp_var.cf{2}.cf{1}.lengthScale);
%         fprintf('PER magnSigma2=%4.3f, l=%4.3f, per=%4.3f \n',...
%             gp_var.cf{2}.cf{2}.magnSigma2, gp_var.cf{2}.cf{2}.lengthScale, gp_var.cf{2}.cf{2}.period);
%         fprintf('SE3 magnSigma2=%4.3f, l=%4.3f \n',...
%             gp_var.cf{3}.magnSigma2, gp_var.cf{3}.lengthScale);
%     end
%     figure(1000); plot(-nll);
%     figure(1001); plot(RMSE);
%     end
    end
end
delete(POOL)

keySet = {'SE','LIN','PER'};
valueSet = {se_init(x,y),lin_init(),per_init(x,y)};
base_kernels=containers.Map(keySet,valueSet);