addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
addpath(genpath('/homes/hkim/Documents/gpml'));
%x=h5read('PPdata_full.h5','/Xtrain');
%y=h5read('PPdata_full.h5','/ytrain');
[n, D] = size(x);
m=40;
X_u=datasample(x,m,1,'Replace',false); %each row specifies coordinates of an inducing point. here we randomly sample m data points
lik = lik_gaussian;
gpcf = gpcf_sexp('lengthScale', ones(1,D), 'magnSigma2', 0.1);
gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf', gpcf,'X_u', X_u); %var_gp

opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','iter','MaxIter',1000);
gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);

%w=gp_pak(gp_var,'covariance+likelihood+inducing'); %params that will be optimised
%[w,~,~,flag]=minimize_stuff(w,@gp_eg,-1000,gp_var,x,y);
%gp_var=gp_unpak(gp_var,w,'covariance+likelihood+inducing');

[~,nll]=gp_e([],gp_var,x,y);