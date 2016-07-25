addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
%addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
parallel=0;
if parallel
    num_workers=10;
    POOL=parpool('local',num_workers);
end

%%% load and set up data %%%
load solar.mat

meanX=mean(X); meany=mean(y);
stdX=std(X); stdy=std(y);

y=(y-meany)/stdy;

[n, D] = size(X);

%%%%%%%%%%%%%%%%%%%
ind=1;
if parallel
    numiter=10;
else numiter=1;
end
nll_table=zeros(numiter,6);
naive_nld_table=zeros(numiter,6);
rff_nld_table=zeros(numiter,6);
cg_obj_table=zeros(numiter,6);
pcg_obj_table=zeros(numiter,6);
pcg_objf_table=zeros(numiter,6);
pcg_objp_table=zeros(numiter,6);

if ~parallel
for m=[10,20,40,80,160,320]
fprintf('m=%d \n',m);
i=1;
%parfor i=1:10
rng(i); fprintf('rng=%d \n',i);
warning('off','all');
%t=getCurrentTask(); k=t.ID;
%filename=['experiment_results/mauna10m',num2str(k),'.txt'];
%fileID=fopen(filename,'at');


%%% Initialise gp_var %%%
per=1;
pl=prior_gaussian('s2',0.5);
lik=lik_gaussian();
gpcf_se1=gpcf_sexp();
gpcf_se2=gpcf_sexp();
gpcf_per = gpcf_periodic('period',per,'lengthScale_sexp_prior',prior_t(),'period_prior',prior_logunif());
gpcf_se=gpcf_sum('cf',{gpcf_se1,gpcf_se2});
gpcf=gpcf_prod('cf',{gpcf_per,gpcf_se});

%%% Optimise gp_var %%%
[~,X_u]=kmeans(X,m); %inducing pts initialised by Kmeans++
gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', X_u); 
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','iter','MaxIter',50);
tic;
gp_var=gp_optim(gp_var,X,y,'opt',opt,'optimf',@fminscg);
time=toc;
fprintf('Time taken for optimising VAR: %4.2f s \n',time);
[~,nll]=gp_e([],gp_var,X,y);
signal_var=gp_var.lik.sigma2;

%%% Extract inducing points from VAR %%%
xu=gp_var.X_u;

%%% Compute UB to NLD %%%
tic;
K_mn=gp_cov(gp_var,xu,X); K_mm=gp_trcov(gp_var,xu);
L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
L=L_mm'\K_mn; %L'*L=K_hat=K_mn'*(K_mm\K_mn)
A=L*L'+signal_var*eye(m);
L_naive=chol(A);
time=toc;
fprintf('Time taken for computing L_naive: %4.4f s \n',time);
naive_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_naive)));
K_naive=L'*L;

%%% Compute RFF and use as UB to NLD %%%
l1=gp_var.cf{1}.cf{2}.cf{1}.lengthScale; sf1=gp_var.cf{1}.cf{2}.cf{1}.magnSigma2;
l2=gp_var.cf{1}.cf{2}.cf{2}.lengthScale; sf2=gp_var.cf{1}.cf{2}.cf{2}.magnSigma2;
lper=gp_var.cf{1}.cf{1}.lengthScale; sfper=gp_var.cf{1}.cf{1}.magnSigma2; per=gp_var.cf{1}.cf{1}.period;
tic;
phi=solarRFF(X,m,l1,sf1,l2,sf2,lper,per,sfper);
Ar=phi*phi'+signal_var*eye(m);
L_rff=chol(Ar);
time=toc;
fprintf('Time taken for computing L_rff: %4.4f s \n',time);
rff_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_rff)));

%%% Compute UB to NIP %%%
[K,C]=gp_trcov(gp_var,X);
tic;
[~,~,~,~,cg_resvec,cg_obj]=cgs_obj(C,y,[],m);
time=toc;
fprintf('Time taken for CG: %4.4f s \n',time);
K_fic=K_naive+diag(diag(K)-diag(K_naive));
dinv=1./(diag(K)-diag(K_naive)+signal_var);
Dinv=diag(dinv); %D=diag(K-K_naive)+signal_var*eye(n)
Af=L*Dinv*L'+eye(m);
%function handle which gives (K_naive+signal_var*eye(n))\x
myfun = @(w) (w-L'*(A\(L*w)))/signal_var;
%function handle which gives (K_fic+signal_var*eye(n))\x
myfunf = @(w) (w-L'*(Af\(L*(w.*dinv)))).*dinv;
%funcion handle which gives (K_pic+signal_var*eye(n))\x
[Kb,invfun]=blockdiag(K-K_naive,m,signal_var); %invfun is fhandle which gives (Kb+signal_var*eye(n))\w
K_pic=K_naive+Kb;
tic;
Ap=L*invfun(L')+eye(m);
time=toc;
fprintf('Time taken for Ap: %4.4f s \n',time);
myfunp = @(w) invfun(w-L'*(Ap\(L*invfun(w))));
tic; [~,~,~,~,pcg_resvec,pcg_obj]=cgs_obj(C,y,[],m,myfun); time=toc;
fprintf('Time taken for PCG Nystrom: %4.4f s \n',time);
tic; [~,~,~,~,pcg_resvecf,pcg_objf]=cgs_obj(C,y,[],m,myfunf); time=toc;
fprintf('Time taken for PCG FIC: %4.4f s \n',time);
tic; [~,~,~,~,pcg_resvecp,pcg_objp]=cgs_obj(C,y,[],m,myfunp); time=toc;
fprintf('Time taken for PCG PIC: %4.4f s \n',time);
cg_obj=minsofar(cg_obj);
pcg_obj=minsofar(pcg_obj);
pcg_objf=minsofar(pcg_objf);
pcg_objp=minsofar(pcg_objp);

%gather all results
nll_table(ind)=nll;
naive_nld_table(ind)=naive_nld;
rff_nld_table(ind)=rff_nld;
cg_obj_table(ind)=cg_obj(end);
pcg_obj_table(ind)=pcg_obj(end);
pcg_objf_table(ind)=pcg_objf(end);
pcg_objp_table(ind)=pcg_objp(end);

%fprintf('worker %d done \n',i);
%end

ind=ind+1;
end

else
for m=[10,20,40,80,160,320]
fprintf('m=%d \n',m);
parfor i=1:10
rng(i);
warning('off','all');
%t=getCurrentTask(); k=t.ID;
%filename=['experiment_results/mauna10m',num2str(k),'.txt'];
%fileID=fopen(filename,'at');


%%% Initialise gp_var %%%
per=1;
pl=prior_gaussian('s2',0.5);
%pl=prior_t();
lik=lik_gaussian();
gpcf_se1=gpcf_sexp('lengthScale_prior',pl);
gpcf_se2=gpcf_sexp('lengthScale_prior',pl);
gpcf_per = gpcf_periodic('period',per,'lengthScale_sexp_prior',prior_t());
gpcf_se=gpcf_sum('cf',{gpcf_se1,gpcf_se2});
gpcf=gpcf_prod('cf',{gpcf_per,gpcf_se});

%%% Optimise gp_var %%%
[~,X_u]=kmeans(X,m); %inducing pts initialised by Kmeans++
gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', X_u); 
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','off','MaxIter',1000);
gp_var=gp_optim(gp_var,X,y,'opt',opt,'optimf',@fminscg);
[~,nll]=gp_e([],gp_var,X,y);
signal_var=gp_var.lik.sigma2;

%%% Extract inducing points from VAR %%%
xu=gp_var.X_u;

%%% Compute UB to NLD %%%
K_mn=gp_cov(gp_var,xu,X); K_mm=gp_trcov(gp_var,xu);
L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
L=L_mm'\K_mn; %L'*L=K_hat=K_mn'*(K_mm\K_mn)
A=L*L'+signal_var*eye(m);
L_naive=chol(A);
naive_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_naive)));
K_naive=L'*L;

%%% Compute RFF and use as UB to NLD %%%
l1=gp_var.cf{1}.cf{2}.cf{1}.lengthScale; sf1=gp_var.cf{1}.cf{2}.cf{1}.magnSigma2;
l2=gp_var.cf{1}.cf{2}.cf{2}.lengthScale; sf2=gp_var.cf{1}.cf{2}.cf{2}.magnSigma2;
lper=gp_var.cf{1}.cf{1}.lengthScale; sfper=gp_var.cf{1}.cf{1}.magnSigma2; per=gp_var.cf{1}.cf{1}.period;
phi=solarRFF(X,m,l1,sf1,l2,sf2,lper,per,sfper);
Ar=phi*phi'+signal_var*eye(m);
L_rff=chol(Ar);
rff_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_rff)));

%%% Compute UB to NIP %%%
[K,C]=gp_trcov(gp_var,X);
[~,~,~,~,cg_resvec,cg_obj]=cgs_obj(C,y,[],m);
K_fic=K_naive+diag(diag(K)-diag(K_naive));
dinv=1./(diag(K)-diag(K_naive)+signal_var);
Dinv=diag(dinv); %D=diag(K-K_naive)+signal_var*eye(n)
Af=L*Dinv*L'+eye(m);
%function handle which gives (K_naive+signal_var*eye(n))\x
myfun = @(w) (w-L'*(A\(L*w)))/signal_var;
%function handle which gives (K_fic+signal_var*eye(n))\x
myfunf = @(w) (w-L'*(Af\(L*(w.*dinv)))).*dinv;
%funcion handle which gives (K_pic+signal_var*eye(n))\x
[Kb,invfun]=blockdiag(K-K_naive,m,signal_var); %invfun is fhandle which gives (Kb+signal_var*eye(n))\w
K_pic=K_naive+Kb;
Ap=L*invfun(L')+eye(m);
myfunp = @(w) invfun(w-L'*(Ap\(L*invfun(w))));
[~,~,~,~,pcg_resvec,pcg_obj]=cgs_obj(C,y,[],m,myfun);
[~,~,~,~,pcg_resvecf,pcg_objf]=cgs_obj(C,y,[],m,myfunf);
[~,~,~,~,pcg_resvecp,pcg_objp]=cgs_obj(C,y,[],m,myfunp); 
cg_obj=minsofar(cg_obj);
pcg_obj=minsofar(pcg_obj);
pcg_objf=minsofar(pcg_objf);
pcg_objp=minsofar(pcg_objp);

%gather all results
nll_table(i,ind)=nll;
naive_nld_table(i,ind)=naive_nld;
rff_nld_table(i,ind)=rff_nld;
cg_obj_table(i,ind)=cg_obj(end);
pcg_obj_table(i,ind)=pcg_obj(end);
pcg_objf_table(i,ind)=pcg_objf(end);
pcg_objp_table(i,ind)=pcg_objp(end);

fprintf('worker %d done \n',i);
end

ind=ind+1;
end
delete(POOL)
end