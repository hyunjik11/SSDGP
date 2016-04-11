addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
%addpath(genpath('/Users/hyunjik11/Documents/Machine_Learning/MLCoursework/GaussianProcesses/gpml-matlab-v3.5-2014-12-08'));
%num_workers=10;
%POOL=parpool('local',num_workers);
% % Load the data
warning('off','all');
load mauna.txt
z = mauna(:,2) ~= -99.99;% get rid of missing data
x = mauna(z,1); y = mauna(z,2); 
x_mean=mean(x); x_std=std(x);
x = (x-mean(x))/x_std; %normalise x;
signal_var=0.1; l=1; sigmaRBF2=1; alpha=2; per=1; coeffSigma2=1;
sigmaRBF=sqrt(sigmaRBF2);
lik = lik_gaussian('sigma2', signal_var);
gpcf_se1 = gpcf_sexp('lengthScale', l, 'magnSigma2',sigmaRBF2); 
gpcf_lin=gpcf_linear('coeffSigma2',coeffSigma2);
gpcf1=gpcf_prod('cf',{gpcf_se1,gpcf_lin});
gpcf_se2 = gpcf_sexp('lengthScale', l, 'magnSigma2', sigmaRBF2);
gpcf_per = gpcf_periodic('lengthScale',l,'period',per,'magnSigma2',sigmaRBF2);
gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
gpcf_se3 = gpcf_sexp('lengthScale', l, 'magnSigma2', sigmaRBF2); 
gpcf_rat = gpcf_rq('lengthScale',l,'magnSigma2',sigmaRBF2,'alpha',alpha); 
gpcf3=gpcf_prod('cf',{gpcf_se3,gpcf_rat});
gp=gp_set('lik',lik,'cf',{gpcf1,gpcf2,gpcf3});
[K,~]=gp_trcov(gp,x);
for m=[100,200,400,800,1600,3200,6400,12800]
z1=randSE(l,m);
b1=2*pi*rand(m,1);
phi_se1=RFF1(x,z1,b1,sigmaRBF);
phi_lin=lin_feat(x,m,coeffSigma2);
idx1=randsample(m^2,m);
phi1=prod_feat(phi_se1,phi_lin,idx1);

half_m=m/2;
z2=randSE(l,half_m);
z3=randPER(per,l,half_m);
phi2=RFFprod(x,z2,z3,sigmaRBF2,sigmaRBF2);
 
z4=randSE(l,half_m);
z5=randRQ(alpha,l,half_m);
phi3=RFFprod(x,z4,z5,sigmaRBF2,sigmaRBF2);

idx2=randsample(2*m,m);
phi_temp=sum_feat(phi1,phi2,idx2);

idx3=randsample(2*m,m);
phi=sum_feat(phi_temp,phi3,idx3);
fprintf('frob=%4.2f\n',norm(phi'*phi-K,'fro'));
end
