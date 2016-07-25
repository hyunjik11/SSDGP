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
gpcf= gpcf_sexp('lengthScale_prior',pl); 
%gpcf_lin=gpcf_linear('coeffSigma2',cs2);

gp=gp_set('lik',lik,'cf',gpcf);
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
