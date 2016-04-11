addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
%addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
num_workers=10;
POOL=parpool('local',num_workers);
% % Load the data
%x=h5read('PPdata.h5','/Xtrain');
%y=h5read('PPdata.h5','/ytrain');
 x=h5read('PPdata_full.h5','/Xtrain');
 y=h5read('PPdata_full.h5','/ytrain');
%% PP500 hyperparams
%x=x(1:500,:); y=y(1:500); %only use 500 pts for faster computation.
% length_scale=[2.0368 3.0397 5.7816 6.9119];
% sigma_RBF2=0.7596;
% signal_var=0.0599;
half_logdet=-15816;
half_innerprod=4784.0;
%% PPfull hyperparams
length_scale=[1.3978 0.0028 2.8966 7.5565];
sigma_RBF2=0.8333; 
signal_var=0.0195;
[n, D] = size(x);


% Now we will use the variational sparse approximation.

% First we create the GP structure. Notice here that if we do
% not explicitly set the priors for the covariance function
% parameters they are given a uniform prior.
lik = lik_gaussian('sigma2', signal_var);
gpcf = gpcf_sexp('lengthScale', length_scale, 'magnSigma2', sigma_RBF2);
gp=gp_set('lik',lik,'cf',gpcf); %exact gp
[K,C]=gp_trcov(gp,x);

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
mean_var_nll=zeros(6,1); std_var_nll=zeros(6,1);
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
    var_nll_values=zeros(10,1);
    
    parfor i=1:10
        warning('off','all');
        X_u=datasample(x,m,1,'Replace',false); %each row specifies coordinates of an inducing point. here we randomly sample m data points
        lik = lik_gaussian('sigma2', 0.2^2);
        gpcf = gpcf_sexp('lengthScale', ones(1,D), 'magnSigma2', 0.2^2);
        gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf', gpcf,'X_u', X_u); %var_gp
        opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','off');
        gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
        [~,nll]=gp_e([],gp_var,x,y);
        var_nll_values(i)=nll;
        xu=gp_var.X_u;
        [K_big,~]=gp_trcov(gp,[x;xu]);
        K_mn=K_big(n+1:n+m,1:n); K_mm=K_big(n+1:n+m,n+1:n+m);
        L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
        L=L_mm'\K_mn; %L'*L=K_hat=K_mn'*(K_mm\K_mn)
%         idx=randsample(n,m);
%         K_mn=K(idx,:); K_mm=K(idx,idx);
%         L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
%         L=L_mm'\K_mn; %L'*L=K_hat=K_mn'*(K_mm\K_mn)
        K_naive=L'*L;
        K_fic=K_naive+diag(diag(K-K_naive));
        K_pic=K_naive+blockdiag(K-K_naive,m);
        Z=randn(m/2,D);
        phi=SEard_RFF2(x,length_scale,sqrt(sigma_RBF2),Z);
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
    mean_var_nll(k)=mean(var_nll_values); std_var_nll(k)=std(var_nll_values);
    k=k+1;
	fprintf('m=%d done \n',m);
end
clear K U S V K_big K_svd K_mn K_mm L_mm L K_naive K_fic K_pic K_rff L_rff L_fic L_pic L_naive temp_rff temp_fic temp_pic temp_naive
save('ppfull_var_indpts_workspace.mat');
delete(POOL);
% k=1;
% mean_nll=zeros(6,1); mean_length_scale1=zeros(6,1);
% mean_sigmaRBF2=zeros(6,1); mean_signal_var=zeros(6,1);
% std_nll=zeros(6,1); std_length_scale1=zeros(6,1);
% std_sigmaRBF2=zeros(6,1); std_signal_var=zeros(6,1);
% for m=[100,200,400,800,1600,3200]
%     nll_values=zeros(10,1);
%     length_scale1_values=zeros(10,1);
%     sigmaRBF2_values=zeros(10,1);
%     signal_var_values=zeros(10,1);
%     parfor i=1:10
%         X_u=datasample(x,m,1,'Replace',false); %each row specifies coordinates of an inducing point. here we randomly sample m data points
%         gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf', gpcf,'X_u', X_u); %var_gp
% if 1==0
% [K,C]=gp_trcov(gp,x);
% %tmp=C\y;
% %fprintf('innerprod/2=%4.2f \n',y'*tmp); %evaluate y'*inv(K+s^2I)*y
% k=1;
% fic_ld_means=zeros(6,1); fic_ld_stds=zeros(6,1);
% fic_ip_means=zeros(6,1); fic_ip_stds=zeros(6,1);
% for m=[100,200,400,800,1600,3200] %number of inducing pts.
% ld_values=zeros(10,1);
% ip_values=zeros(10,1);
% for i=1:10
% idx=randsample(n,m);
% K_mn=K(idx,:); K_mm=K(idx,idx);
% L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
% L=L_mm'\K_mn; %L'*L=K_hat=K_mn'*(K_mm\K_mn)
% K_hat=L'*L;
% K_hat=K_hat+diag(diag(K-K_hat));
% %K_hat=K_hat+blockdiag(K-K_hat,m);
% L=chol(K_hat+signal_var*eye(n));
% temp=L'\y;
% ip_value=temp'*temp/2;
% ld_value=sum(log(diag(L)));
% ip_values(i)=ip_value;
% ld_values(i)=ld_value;
% end
% ip_mean=mean(ip_values); ip_std=std(ip_values);
% ld_mean=mean(ld_values); ld_std=std(ld_values);
% fic_ip_means(k)=ip_mean; fic_ip_stds(k)=ip_std;
% fic_ld_means(k)=ld_mean; fic_ld_stds(k)=ld_std; 
% k=k+1;
% fprintf('m=%d, ld_mean=%4.4f, ld_std=%4.4f \n',m,ld_mean,ld_std)
% fprintf('m=%d, ip_mean=%4.4f, ip_std=%4.4f \n',m,ip_mean,ip_std)
% end
% end
% Next we initialize the inducing inputs and set them in GP
% structure. We have to give a prior for the inducing inputs also,
% if we want to optimize them


% -----------------------------
% --- Conduct the inference ---

% Then we can conduct the inference. We can now optimize i) only
% the parameters, ii) both the parameters and the inducing inputs,
% or iii) only the inducing inputs. Which option is used is defined
% by a string that is given to the gp_pak, gp_unpak, gp_e and gp_g
% functions. The strings for the different options are:
% 'covariance+likelihood' (i), 'covariance+likelihood+inducing' (ii),
% 'inducing' (iii).
%

% Now you can choose, if you want to optimize only parameters or
% optimize simultaneously parameters and inducing inputs. Note that
% the inducing inputs are not transformed through logarithm when
% packed

% optimize parameters and inducing inputs
% gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
% optimize only parameters
%gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');           

%opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','off','MaxIter',1000);%,'Display','off');
% Optimize with the quasi-Newton method
%gp=gp_optim(gp,x,y,'opt',opt);
%gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg); %can also use @fminlbfgs,@fminunc
% Set the options for the optimization
%for m=1:10
%xm=x(1:10*m,:); ym=y(1:10*m);
%fprintf('m=%d,-l=%2.4f \n',10*m,gp_e([],gp,xm,ym));
%end
%[temp,nll]=gp_e([],gp_var,x,y);
%nll_values(i)=nll; length_scale1_values(i)=gp_var.cf{1}.lengthScale(1);
%sigmaRBF2_values(i)=gp_var.cf{1}.magnSigma2; signal_var_values(i)=gp_var.lik.sigma2;
%fprintf('-l=%2.4f;',nll);
%fprintf('length_scale=[');
%fprintf('%s',num2str(gp.cf{1}.lengthScale));
%fprintf('];sigma_RBF2=%2.4f;signal_var=%2.4f \n',gp.cf{1}.magnSigma2,gp.lik.sigma2);
%fprintf('-l=%2.4f;',gp_e([],gp_var,x,y));
%fprintf('length_scale=[');
%fprintf('%s',num2str(gp_var.cf{1}.lengthScale));
%fprintf('];sigma_RBF2=%2.4f;signal_var=%2.4f \n',gp_var.cf{1}.magnSigma2,gp_var.lik.sigma2);
%     end
% mean_nll(k)=mean(nll_values); mean_length_scale1(k)=mean(length_scale1_values);
% mean_sigmaRBF2(k)=mean(sigmaRBF2_values); mean_signal_var(k)=mean(signal_var_values);
% std_nll(k)=std(nll_values); std_length_scale1(k)=std(length_scale1_values);
% std_sigmaRBF2(k)=std(sigmaRBF2_values); std_signal_var(k)=std(signal_var_values);
% k=k+1;
% end

