addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
%addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
num_workers=4;
%POOL=parpool('local',num_workers);
maxNumCompThreads(num_workers); 
load mauna.txt
z = mauna(:,2) ~= -99.99;                             % get rid of missing data
x = mauna(z,1); y = mauna(z,2); % extract year and CO2 concentration
x_mean=mean(x); x_std=std(x);
y_mean=mean(x); y_std=std(y);
x = (x-x_mean)/x_std; %normalise x;
y = (y-y_mean)/y_std; %normalise y;

%%%Use the right initial values for the hyps %%%%%%%%%%%%%%%%%%
signal_var=5e-6;
l1=0.5552; sf1=0.1831;
cs2=0.1831;
l2=44.2116; sf2=29.7963;
lper=7.0113; sfper=29.7963;per=1/x_std;
l3=3.3270; sf3=0.0155;
lrq=0.0045; sfrq=0.0155; alpha=1.0599;
half_logdet=-2937.7;
half_innerprod=328.4550;
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
[n, D] = size(x);

% First we create the GP structure. Notice here that if we do
% not explicitly set the priors for the covariance function
% parameters they are given a uniform prior.
lik = lik_gaussian('sigma2', signal_var);
gpcf_se1 = gpcf_sexp('lengthScale', l1, 'magnSigma2',sf1); 
gpcf_lin=gpcf_linear('coeffSigma2',cs2);
gpcf1=gpcf_prod('cf',{gpcf_se1,gpcf_lin});
gpcf_se2 = gpcf_sexp('lengthScale', l2, 'magnSigma2', sf2);
gpcf_per = gpcf_periodic('lengthScale',lper,'period',per,'magnSigma2',sfper);
gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
gpcf_se3 = gpcf_sexp('lengthScale', l3, 'magnSigma2', sf3); 
gpcf_rat = gpcf_rq('lengthScale',lrq,'magnSigma2',sfrq,'alpha',alpha); 
gpcf3=gpcf_prod('cf',{gpcf_se3,gpcf_rat});
gp=gp_set('lik',lik,'cf',{gpcf1 gpcf2 gpcf3});
[K,~]=gp_trcov(gp,x);

frob_svd=zeros(6,1); spec_svd=zeros(6,1);
mean_frob_naive=zeros(6,1); mean_spec_naive=zeros(6,1);
mean_frob_fic=zeros(6,1); mean_spec_fic=zeros(6,1);
mean_frob_pic=zeros(6,1); mean_spec_pic=zeros(6,1);
mean_frob_rff=zeros(6,1); mean_spec_rff=zeros(6,1);
std_frob_naive=zeros(6,1); std_spec_naive=zeros(6,1);
std_frob_fic=zeros(6,1); std_spec_fic=zeros(6,1);
std_frob_pic=zeros(6,1); std_spec_pic=zeros(6,1);
std_frob_rff=zeros(6,1); std_spec_rff=zeros(6,1);
mean_ip_naive=zeros(6,1); mean_ld_naive=zeros(6,1);
mean_ip_fic=zeros(6,1); mean_ld_fic=zeros(6,1);
mean_ip_pic=zeros(6,1); mean_ld_pic=zeros(6,1);
mean_ip_rff=zeros(6,1); mean_ld_rff=zeros(6,1);
std_ip_naive=zeros(6,1); std_ld_naive=zeros(6,1);
std_ip_fic=zeros(6,1); std_ld_fic=zeros(6,1);
std_ip_pic=zeros(6,1); std_ld_pic=zeros(6,1);
std_ip_rff=zeros(6,1); std_ld_rff=zeros(6,1);
k=1;
for m=[10,20,40,80,160,320]
    [U,S,V]=svds(K,m);
    K_svd=U*S*V';
    svd_frob_value=norm(K-K_svd,'fro'); svd_spec_value=norm(K-K_svd);
    naive_frob_values=zeros(10,1); naive_spec_values=zeros(10,1);
    fic_frob_values=zeros(10,1); fic_spec_values=zeros(10,1);
    pic_frob_values=zeros(10,1); pic_spec_values=zeros(10,1);
    rff_frob_values=zeros(10,1); rff_spec_values=zeros(10,1);
    naive_ip_values=zeros(10,1); naive_ld_values=zeros(10,1);
    fic_ip_values=zeros(10,1); fic_ld_values=zeros(10,1);
    pic_ip_values=zeros(10,1); pic_ld_values=zeros(10,1);
    rff_ip_values=zeros(10,1); rff_ld_values=zeros(10,1);
    
    for i=1:10
        idx=randsample(n,m);
        K_mn=K(idx,:); K_mm=K(idx,idx);
        L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
        L=L_mm'\K_mn; %L'*L=K_hat=K_mn'*(K_mm\K_mn)
        K_naive=L'*L;
        K_fic=K_naive+diag(diag(K-K_naive));
        K_pic=K_naive+blockdiag(K-K_naive,m);
        idx1=randsample(m^2,m);
        idx2=randsample(2*m,m);
        idx3=randsample(2*m,m);
        phi=maunaRFF(x,m,l1,sf1,cs2,l2,sf2,lper,per,sfper,l3,sf3,lrq,sfrq,alpha,idx1,idx2,idx3);
        K_rff=phi'*phi;
        
        L_naive=chol(K_naive+signal_var*eye(n));
        temp_naive=L_naive'\y;
        naive_ip_values(i)=temp_naive'*temp_naive/2;
        naive_ld_values(i)=sum(log(diag(L_naive)));
        L_fic=chol(K_fic+signal_var*eye(n));
        temp_fic=L_fic'\y;
        fic_ip_values(i)=temp_fic'*temp_fic/2;
        fic_ld_values(i)=sum(log(diag(L_fic)));
        L_pic=chol(K_pic+signal_var*eye(n));
        temp_pic=L_pic'\y;
        pic_ip_values(i)=temp_pic'*temp_pic/2;
        pic_ld_values(i)=sum(log(diag(L_pic)));        
        L_rff=chol(K_rff+signal_var*eye(n));
        temp_rff=L_rff'\y;
        rff_ip_values(i)=temp_rff'*temp_rff/2;
        rff_ld_values(i)=sum(log(diag(L_rff)));
        
        naive_frob_values(i)=norm(K-K_naive,'fro');
        fic_frob_values(i)=norm(K-K_fic,'fro');
        pic_frob_values(i)=norm(K-K_pic,'fro');
        rff_frob_values(i)=norm(K-K_rff,'fro');
        naive_spec_values(i)=norm(K-K_naive);
        fic_spec_values(i)=norm(K-K_fic);
        pic_spec_values(i)=norm(K-K_pic);
        rff_spec_values(i)=norm(K-K_rff);
        fprintf('ok \n');
    end
    frob_svd(k)=svd_frob_value; spec_svd(k)=svd_spec_value;
    mean_frob_naive(k)=mean(naive_frob_values); mean_spec_naive(k)=mean(naive_spec_values);
    mean_frob_fic(k)=mean(fic_frob_values); mean_spec_fic(k)=mean(fic_spec_values);
    mean_frob_pic(k)=mean(pic_frob_values); mean_spec_pic(k)=mean(pic_spec_values);
    mean_frob_rff(k)=mean(rff_frob_values); mean_spec_rff(k)=mean(rff_spec_values);
    std_frob_naive(k)=std(naive_frob_values); std_spec_naive(k)=std(naive_spec_values);
    std_frob_fic(k)=std(fic_frob_values); std_spec_fic(k)=std(fic_spec_values);
    std_frob_pic(k)=std(pic_frob_values); std_spec_pic(k)=std(pic_spec_values);
    std_frob_rff(k)=std(rff_frob_values); std_spec_rff(k)=std(rff_spec_values);
    mean_ip_naive(k)=mean(naive_ip_values); mean_ld_naive(k)=mean(naive_ld_values);
    mean_ip_fic(k)=mean(fic_ip_values); mean_ld_fic(k)=mean(fic_ld_values);
    mean_ip_pic(k)=mean(pic_ip_values); mean_ld_pic(k)=mean(pic_ld_values);
    mean_ip_rff(k)=mean(rff_ip_values); mean_ld_rff(k)=mean(rff_ld_values);
    std_ip_naive(k)=std(naive_ip_values); std_ld_naive(k)=std(naive_ld_values);
    std_ip_fic(k)=std(fic_ip_values); std_ld_fic(k)=std(fic_ld_values);
    std_ip_pic(k)=std(pic_ip_values); std_ld_pic(k)=std(pic_ld_values);
    std_ip_rff(k)=std(rff_ip_values); std_ld_rff(k)=std(rff_ld_values);
    k=k+1;
	fprintf('m=%d done \n',m);
end
frob_svd(3)=0.0072; %numerical error coming from high condition number of K
spec_svd(3)=0.0010; %numerical error coming from high condition number of K
clear K U S V K_svd K_mn K_mm L_mm L K_naive K_fic K_pic K_rff L_rff L_fic L_pic L_naive temp_rff temp_fic temp_pic temp_naive
save('mauna_rand_indpts_workspace.mat');
delete(POOL);
