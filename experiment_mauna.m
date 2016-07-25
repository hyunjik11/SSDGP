addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
%addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
parallel=1;
if parallel
    num_workers=10;
    POOL=parpool('local',num_workers);
end

%%% load and set up data %%%
load mauna.txt
z = mauna(:,2) ~= -99.99; % get rid of missing data
x = mauna(z,1); y = mauna(z,2); % extract year and CO2 concentration
x_mean=mean(x); x_std=std(x);
y_mean=mean(y); y_std=std(y);
x = (x-x_mean)/x_std; %normalise x;
y = (y-y_mean)/y_std; %normalise y;
[n,D]=size(x);

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
l1=0.10;sf1=0.1;cs2=1.00;l2=1.00;sf2=0.1;lper=1.00;sfper=0.1;l3=1.00;sf3=0.1;
per=1/x_std; signal_var=0.1;
pl=prior_gaussian('s2',0.5);
lik = lik_gaussian('sigma2', signal_var);
gpcf_se1 = gpcf_sexp('lengthScale', l1, 'magnSigma2',sf1,'lengthScale_prior',pl); 
gpcf_lin=gpcf_linear('coeffSigma2',cs2);
gpcf1=gpcf_prod('cf',{gpcf_se1,gpcf_lin});
gpcf_se2 = gpcf_sexp('lengthScale', l2, 'magnSigma2', sf2,'lengthScale_prior',pl);
gpcf_per = gpcf_periodic('lengthScale',lper,'period',per,'magnSigma2',sfper,'lengthScale_prior',pl);
gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
gpcf3 = gpcf_sexp('lengthScale', l3, 'magnSigma2', sf3,'lengthScale_prior',pl); 


%%% Optimise gp_var %%%
[~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3},'X_u', X_u); 
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','iter','MaxIter',1000);
tic;
gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
time=toc;
fprintf('Time taken for optimising VAR: %4.2f s \n',time);
[~,nll]=gp_e([],gp_var,x,y);
signal_var=gp_var.lik.sigma2;

%%% Extract inducing points from VAR %%%
xu=gp_var.X_u;

%%% Compute UB to NLD %%%
tic;
K_mn=gp_cov(gp_var,xu,x); K_mm=gp_trcov(gp_var,xu);
L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
L=L_mm'\K_mn; %L'*L=K_hat=K_mn'*(K_mm\K_mn)
A=L*L'+signal_var*eye(m);
L_naive=chol(A);
time=toc;
fprintf('Time taken for computing L_naive: %4.4f s \n',time);
naive_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_naive)));
K_naive=L'*L;

%%% Compute RFF and use as UB to NLD %%%
l1=gp_var.cf{1}.cf{1}.lengthScale; sf1=gp_var.cf{1}.cf{1}.magnSigma2;
cs2=gp_var.cf{1}.cf{2}.coeffSigma2;
l2=gp_var.cf{2}.cf{1}.lengthScale; sf2=gp_var.cf{2}.cf{1}.magnSigma2;
lper=gp_var.cf{2}.cf{2}.lengthScale; sfper=gp_var.cf{2}.cf{2}.magnSigma2; per=gp_var.cf{2}.cf{2}.period;
l3=gp_var.cf{3}.lengthScale; sf3=gp_var.cf{3}.magnSigma2;
tic;
idx1=randsample(m^2,m);
idx2=randsample(2*m,m);
idx3=randsample(2*m,m);
phi=maunaRFF(x,m,l1,sf1,cs2,l2,sf2,lper,per,sfper,l3,sf3,idx1,idx2,idx3);
Ar=phi*phi'+signal_var*eye(m);
L_rff=chol(Ar);
time=toc;
fprintf('Time taken for computing L_rff: %4.4f s \n',time);
rff_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_rff)));

%%% Compute UB to NIP %%%
[K,C]=gp_trcov(gp_var,x);
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
l1=0.10;sf1=0.1;cs2=1.00;l2=1.00;sf2=0.1;lper=1.00;sfper=0.1;l3=1.00;sf3=0.1;
per=1/x_std; signal_var=0.1;
%pl=prior_gaussian('s2',0.5);
pl=prior_t();
lik = lik_gaussian('sigma2', signal_var);
gpcf_se1 = gpcf_sexp('lengthScale', l1, 'magnSigma2',sf1,'lengthScale_prior',pl); 
gpcf_lin=gpcf_linear('coeffSigma2',cs2);
gpcf1=gpcf_prod('cf',{gpcf_se1,gpcf_lin});
gpcf_se2 = gpcf_sexp('lengthScale', l2, 'magnSigma2', sf2,'lengthScale_prior',pl);
gpcf_per = gpcf_periodic('lengthScale',lper,'period',per,'magnSigma2',sfper,'lengthScale_prior',pl);
gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
gpcf3 = gpcf_sexp('lengthScale', l3, 'magnSigma2', sf3,'lengthScale_prior',pl); 


%%% Optimise gp_var %%%
[~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3},'X_u', X_u); 
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','off','MaxIter',1000);
gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
[~,nll]=gp_e([],gp_var,x,y);
signal_var=gp_var.lik.sigma2;

%%% Extract inducing points from VAR %%%
xu=gp_var.X_u;

%%% Compute UB to NLD %%%
K_mn=gp_cov(gp_var,xu,x); K_mm=gp_trcov(gp_var,xu);
L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
L=L_mm'\K_mn; %L'*L=K_hat=K_mn'*(K_mm\K_mn)
A=L*L'+signal_var*eye(m);
L_naive=chol(A);
naive_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_naive)));
K_naive=L'*L;

%%% Compute RFF and use as UB to NLD %%%
l1=gp_var.cf{1}.cf{1}.lengthScale; sf1=gp_var.cf{1}.cf{1}.magnSigma2;
cs2=gp_var.cf{1}.cf{2}.coeffSigma2;
l2=gp_var.cf{2}.cf{1}.lengthScale; sf2=gp_var.cf{2}.cf{1}.magnSigma2;
lper=gp_var.cf{2}.cf{2}.lengthScale; sfper=gp_var.cf{2}.cf{2}.magnSigma2; per=gp_var.cf{2}.cf{2}.period;
l3=gp_var.cf{3}.lengthScale; sf3=gp_var.cf{3}.magnSigma2;
idx1=randsample(m^2,m);
idx2=randsample(2*m,m);
idx3=randsample(2*m,m);
phi=maunaRFF(x,m,l1,sf1,cs2,l2,sf2,lper,per,sfper,l3,sf3,idx1,idx2,idx3);
Ar=phi*phi'+signal_var*eye(m);
L_rff=chol(Ar);
rff_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_rff)));

%%% Compute UB to NIP %%%
[K,C]=gp_trcov(gp_var,x);
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



