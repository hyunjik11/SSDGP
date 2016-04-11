%addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
%addpath(genpath('/homes/hkim/Documents/gpml-matlab-v3.6-2015-07-07'));
addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
%addpath(genpath('/Users/hyunjik11/Documents/Machine_Learning/MLCoursework/GaussianProcesses/gpml-matlab-v3.5-2014-12-08'));
%num_workers=10;
%POOL=parpool('local',num_workers);
% % Load the data
load mauna.txt
z = mauna(:,2) ~= -99.99;                             % get rid of missing data
x = mauna(z,1); y = mauna(z,2); % extract year and CO2 concentration
x_mean=mean(x); x_std=std(x);
y_mean=mean(x); y_std=std(y);
x = (x-x_mean)/x_std; %normalise x;
y = (y-y_mean)/y_std; %normalise y;
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
small_set=[0.01,0.1];
large_set=[0.1,1];
huge_set=[1,10];
m=10;
X_u=datasample(x,m,1,'Replace',false); %each row specifies coordinates of an inducing point. here we randomly sample m data points
%signal_var=0.10;l=0.10;sigmaf2=0.10;coeffSigma2=1;alpha=1.1;
%signal_var=0.1;l=1;sigmaf2=1;coeffSigma2=1;alpha=2;
warning('off','all');
s=[2,2,2,2,2,2,2,2,2,2,2];
for ind=1:2^11
[i1,i2,i3,i4,i5,i6,i7,i8,i9,i10,i11]=ind2sub(s,ind);
l1=small_set(i1); sf1=small_set(i2);
cs2=large_set(i3);
l2=huge_set(i4); sf2=small_set(i5);
lper=large_set(i6); sfper=small_set(i7);
l3=huge_set(i8); sf3=huge_set(i9);
lrq=huge_set(i10); sfrq=huge_set(i11);

lik = lik_gaussian('sigma2', 0.1);
gpcf_se1 = gpcf_sexp('lengthScale', l1, 'magnSigma2',sf1); 
gpcf_lin=gpcf_linear('coeffSigma2',cs2);
gpcf1=gpcf_prod('cf',{gpcf_se1,gpcf_lin});
gpcf_se2 = gpcf_sexp('lengthScale', l2, 'magnSigma2', sf2);
gpcf_per = gpcf_periodic('lengthScale',lper,'period',1/x_std,'magnSigma2',sfper);
gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
gpcf_se3 = gpcf_sexp('lengthScale', l3, 'magnSigma2', sf3); 
gpcf_rat = gpcf_rq('lengthScale',lrq,'magnSigma2',sfrq,'alpha',1.1); 
gpcf3=gpcf_prod('cf',{gpcf_se3,gpcf_rat});
gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3},'X_u', X_u); %var_gp%gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
%gp=gp_set('lik',lik,'cf',{gpcf1 gpcf2 gpcf3});
opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','off','MaxIter',1000);
%gp=gp_optim(gp,x,y,'opt',opt);
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
gp_var=gp_optim(gp_var,x,y,'opt',opt);
%[~,nll]=gp_e([],gp,x,y);
[~,nll]=gp_e([],gp_var,x,y);
fprintf('nll=%4.2f,l1=%4.2f,sf1=%4.2f,cs2==%4.2f,l2=%4.2f,sf2=%4.2f,lper=%4.2f,sfper==%4.2f,l3=%4.2f,sf3=%4.2f,lrq=%4.2f,sfrq==%4.2f\n',nll,l1,sf1,cs2,l2,sf2,lper,sfper,l3,sf3,lrq,sfrq);
end
%zz = ((2016+1/24:1/12:2024-1/24)'-x_mean)/x_std;
%[mu,s2]=gp_pred(gp,x,y,zz);
%[mu,s2]=gp_pred(gp_var,x,y,zz);
%f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)]*y_std + y_mean;
%figure();
%fill([zz*x_std+x_mean; flip(zz*x_std+x_mean,1)], f, [7 7 7]/8);  hold on       % show predictions
%plot(x*x_std+x_mean,y*y_std+y_mean,'b.'); plot(zz*x_std+x_mean,mu*y_std+y_mean,'r.');
% end
% end
% end
% end
% end
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
% [~,nll]=gp_e([],gp_var,x,y);
% [K,C]=gp_trcov(gp,x);
% L=chol(C);
% temp=L'\y;
% ip_value=temp'*temp/2;
% ld_value=sum(log(diag(L)));
