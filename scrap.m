%addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
%addpath(genpath('/Users/hyunjik11/Documents/Machine_Learning/MLCoursework/GaussianProcesses/gpml-matlab-v3.5-2014-12-08'));
addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
num_workers=4;
%POOL=parpool('local',num_workers);
% % Load the data
load mauna.txt
z = mauna(:,2) ~= -99.99;% get rid of missing data
x = mauna(z,1); y = mauna(z,2); % extract year and CO2 concentration
x_mean=mean(x); x_std=std(x);
y_mean=mean(x); y_std=std(y);
x = (x-x_mean)/x_std; %normalise x;
y = (y-y_mean)/y_std; %normalise y;
signal_var=0.1;
l1=0.1; sf1=0.01;
cs2=1;
l2=1; sf2=0.1;
lper=1; sfper=0.01;per=1/x_std;
l3=1; sf3=1;
lrq=1; sfrq=10; alpha=2;
m=10;
% small_set=[0.01,0.1];
% large_set=[0.1,1];
% huge_set=[1,10];
% alpha_set=[2,10];
% s=[2,2,2,2,2];%signal_var,l,sf,cs2,alpha
fprintf('m=%d,l1=%4.2f,sf1=%4.2f,cs2==%4.2f,l2=%4.2f,sf2=%4.2f,lper=%4.2f,sfper=%4.2f,l3=%4.2f,sf3=%4.2f,lrq=%4.2f,sfrq=%4.2f\n',m,l1,sf1,cs2,l2,sf2,lper,sfper,l3,sf3,lrq,sfrq);
nll_values=zeros(10,1);
parfor i=1:10
warning('off','all');
%t=getCurrentTask(); k=t.ID;
%filename=['experiment_results/mauna10m',num2str(k),'.txt'];
%fileID=fopen(filename,'at');

lik = lik_gaussian('sigma2', signal_var);
gpcf_se1 = gpcf_sexp('lengthScale', l1, 'magnSigma2',sf1); 
gpcf_lin=gpcf_linear('coeffSigma2',cs2);
gpcf1=gpcf_prod('cf',{gpcf_se1,gpcf_lin});
gpcf_se2 = gpcf_sexp('lengthScale', l2, 'magnSigma2', sf2);
gpcf_per = gpcf_periodic('lengthScale',lper,'period',per,'magnSigma2',sfper);
gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
gpcf_se3 = gpcf_sexp('lengthScale', l3, 'magnSigma2', sf3); 
gpcf_rat = gpcf_rq('lengthScale',lrq,'magnSigma2',sfrq,'alpha',2); 
gpcf3=gpcf_prod('cf',{gpcf_se3,gpcf_rat});
X_u=datasample(x,m,1,'Replace',false);
gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3},'X_u', X_u); %var_gp%gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','off','MaxIter',1000);
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
gp_var=gp_optim(gp_var,x,y,'opt',opt);

% gp=gp_set('lik',lik,'cf',{gpcf1,gpcf2,gpcf3});
% opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','iter','MaxIter',1000);
% gp=gp_optim(gp,x,y,'opt',opt);
%[~,nll]=gp_e([],gp,x,y);
[~,nll]=gp_e([],gp_var,x,y);
% zz = ((2016+1/24:1/12:2024-1/24)'-x_mean)/x_std;
% [mu,s2]=gp_pred(gp_var,x,y,zz);
% f = [mu+2*sqrt(s2); flip(mu-2*sqrt(s2),1)]*y_std + y_mean;
% figure();
% fill([zz*x_std+x_mean; flip(zz*x_std+x_mean,1)], f, [7 7 7]/8);  hold on       % show predictions
% plot(x*x_std+x_mean,y*y_std+y_mean,'b.'); plot(zz*x_std+x_mean,mu*y_std+y_mean,'r.');
nll_values(i)=nll;
%fprintf('nll=%4.2f,l1=%4.2f,sf1=%4.2f,cs2==%4.2f,l2=%4.2f,sf2=%4.2f,lper=%4.2f,sfper=%4.2f,l3=%4.2f,sf3=%4.2f,lrq=%4.2f,sfrq=%4.2f\n',nll,l1,sf1,cs2,l2,sf2,lper,sfper,l3,sf3,lrq,sfrq);
end
fprintf('mean_nll=%4.2f,std_nll=%4.2f',mean(nll_values),std(nll_values));
%delete(POOL)
%[K,~]=gp_trcov(gp,x);
% for m=[10,20,40,80,160,320]
%     idx1=randsample(m^2,m);
%     idx2=randsample(2*m,m);
%     idx3=randsample(2*m,m);
%     phi=maunaRFF(x,m,l1,sf1,cs2,l2,sf2,lper,per,sfper,l3,sf3,lrq,sfrq,alpha,idx1,idx2,idx3);
%     fprintf('frob=%4.2f\n',norm(phi'*phi-K,'fro'));
% end