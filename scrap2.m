addpath(genpath('/homes/hkim/Documents/GPstuff-4.6')); 
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
%scatter(x,y);
%hold on
%plot(x,pred);

%delete(POOL)
