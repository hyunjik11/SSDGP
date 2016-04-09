%addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
%addpath(genpath('/homes/hkim/Documents/gpml-matlab-v3.6-2015-07-07'));
addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
%addpath(genpath('/Users/hyunjik11/Documents/Machine_Learning/MLCoursework/GaussianProcesses/gpml-matlab-v3.5-2014-12-08'));
%num_workers=10;
%POOL=parpool('local',num_workers);
% % Load the data
load mauna.txt
z = mauna(:,2) ~= -99.99;                             % get rid of missing data
x = mauna(z,1); y = mauna(z,2);       % extract year and CO2 concentration

% k1={@covProd, {@covSEiso,@covLINiso}};
% k2={@covProd, {@covSEiso,@covPeriodic}};
% k3={@covProd, {@covSEiso, @covRQiso}};
% covfunc={@covSum, {k1, k2, k3}};      
% hyp.cov=[0 0 0 0 0 0 0 0 0 0 0 0 0];
% hyp.lik=-2;
% [hyp fX i] =minimize(hyp, @gp, -500, @infExact, [], covfunc, @likGauss, x, y-mean(y));
% zz = (2016+1/24:1/12:2024-1/24)';  
% [mu s2] = gp(hyp, @infExact, [], covfunc, @likGauss, x, y-mean(y), zz);
% f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)] + mean(y);
% fill([zz; flipdim(zz,1)], f, [7 7 7]/8); hold on;           % show predictions
% plot(x,y,'b.'); plot(zz,mu+mean(y),'r.');
lik = lik_gaussian('sigma2', 0.1);
gpcf_se1 = gpcf_sexp('lengthScale', 1, 'magnSigma2',1); 
gpcf_lin=gpcf_linear('coeffSigma2',1);
gpcf1=gpcf_prod('cf',{gpcf_se1,gpcf_lin});
gpcf_se2 = gpcf_sexp('lengthScale', 1, 'magnSigma2', 1);
gpcf_per = gpcf_periodic('lengthScale',1,'period',1);
gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
gpcf_se3 = gpcf_sexp('lengthScale', 1, 'magnSigma2', 1); 
gpcf_rat = gpcf_rq('lengthScale',1,'magnSigma2',1,'alpha',2); 
%%%% NOTE FROM SETH: for whatever reason, alpha = 1 above doesn't work.
gpcf3=gpcf_prod('cf',{gpcf_se3,gpcf_rat});
m=80;
X_u=datasample(x,m,1,'Replace',false); %each row specifies coordinates of an inducing point. here we randomly sample m data points
gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3},'X_u', X_u); %var_gp%gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
%gp=gp_set('lik',lik,'cf',{gpcf1 gpcf2 gpcf3});
opt=optimset('TolFun',1e-6,'TolX',1e-6,'Display','iter','MaxIter',1000);
%gp=gp_optim(gp,x,y-mean(y),'opt',opt);
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
gp_var=gp_optim(gp_var,x,y-mean(y),'opt',opt);
zz = (2016+1/24:1/12:2024-1/24)';
%[mu s2]=gp_pred(gp,x,y-mean(y),zz);
[mu,s2]=gp_pred(gp_var,x,y-mean(y),zz);
f = [mu+2*sqrt(s2); flipdim(mu-2*sqrt(s2),1)] + mean(y);
fill([zz; flipdim(zz,1)], f, [7 7 7]/8); hold on;           % show predictions
plot(x,y,'b.'); plot(zz,mu+mean(y),'r.');
% m=10;
% X_u=datasample(x,m,1,'Replace',false); %each row specifies coordinates of an inducing point. here we randomly sample m data points
% gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf', gpcf,'X_u', X_u); %var_gp
% gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
% % optimize only parameters
% %gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');           
% 
% opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','iter','MaxIter',1000);
% % Optimize with the quasi-Newton method
% %gp=gp_optim(gp,x,y,'opt',opt);
% tic()
% gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg); %can also use @fminlbfgs,@fminunc
% toc()
% fprintf('m=%d,-l=%2.4f \n',10*m,gp_e([],gp_var,x,y));
% %end
% [temp,nll]=gp_e([],gp_var,x,y);
% [K,C]=gp_trcov(gp,x);
% L=chol(C);
% temp=L'\y;
% ip_value=temp'*temp/2;
% ld_value=sum(log(diag(L)));
