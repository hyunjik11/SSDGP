addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
%addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
parallel=1;
if parallel
    num_workers=10;
    POOL=parpool('local',num_workers);
end

%%% load and set up data %%%
load solar.mat
x=X;
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
m_values=[10,20,40,80,160,320];
lm=length(m_values);
lb_table=zeros(numiter,lm);
naive_nld_table=zeros(1,lm);
rff_nld_table=zeros(numiter,lm);
cg_obj_table=zeros(1,lm);
pcg_obj_table=zeros(1,lm);
pcg_objf_table=zeros(1,lm);
pcg_objp_table=zeros(1,lm);
nld = zeros(1,lm);
nip = zeros(1,lm);
gp_var_cell=cell(1,10);

if ~parallel
for m=m_values
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
[~,X_u]=kmeans(xw,m); %inducing pts initialised by Kmeans++
gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', X_u); 
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','iter','MaxIter',1000);
tic;
gp_var=gp_optim(gp_var,xw,yw,'opt',opt,'optimf',@fminscg);
time=toc;
fprintf('Time taken for optimising VAR: %4.2f s \n',time);
[~,nll]=gp_e([],gp_var,xw,yw);
signal_var=gp_var.lik.sigma2;

%%% Extract inducing points from VAR %%%
xu=gp_var.X_u;

%%% Compute UB to NLD %%%
tic;
K_mn=gp_cov(gp_var,xu,xw); K_mm=gp_trcov(gp_var,xu);
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
phi=solarRFF(xw,m,l1,sf1,l2,sf2,lper,per,sfper);
Ar=phi*phi'+signal_var*eye(m);
L_rff=chol(Ar);
time=toc;
fprintf('Time taken for computing L_rff: %4.4f s \n',time);
rff_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_rff)));

%%% Compute UB to NIP %%%
[K,C]=gp_trcov(gp_var,xw);
tic;
[~,~,~,~,cg_resvec,cg_obj]=cgs_obj(C,yw,[],m);
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
tic; [~,~,~,~,pcg_resvec,pcg_obj]=cgs_obj(C,yw,[],m,myfun); time=toc;
fprintf('Time taken for PCG Nystrom: %4.4f s \n',time);
tic; [~,~,~,~,pcg_resvecf,pcg_objf]=cgs_obj(C,yw,[],m,myfunf); time=toc;
fprintf('Time taken for PCG FIC: %4.4f s \n',time);
tic; [~,~,~,~,pcg_resvecp,pcg_objp]=cgs_obj(C,yw,[],m,myfunp); time=toc;
fprintf('Time taken for PCG PIC: %4.4f s \n',time);
cg_obj=minsofar(cg_obj);
pcg_obj=minsofar(pcg_obj);
pcg_objf=minsofar(pcg_objf);
pcg_objp=minsofar(pcg_objp);

%gather all results
lb_table(ind)=nll;
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
for m=m_values
fprintf('m=%d \n',m);
parfor i=1:10
rng(i);
warning('off','all');
%t=getCurrentTask(); k=t.ID;
%filename=['experiment_results/mauna10m',num2str(k),'.txt'];
%fileID=fopen(filename,'at');


%%% Initialise gp_var %%%
lik=lik_init(y);
gpcf_se=se_init(x,y);
gpcf_per = per_init(x,y);
gpcf=gpcf_prod('cf',{gpcf_per,gpcf_se});

%%% Optimise gp_var %%%
[~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',gpcf,'X_u', X_u); 
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood+inducing');
opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
[~,nll]=gp_e([],gp_var,x,y);

%%% Record ml and gp_var %%%
lb_table(i,ind)=-nll;
gp_var_cell{i} = gp_var;
fprintf('optim for worker %d done \n',i);
end

%%% Get index of best var LB %%%
[~,maxind]=max(lb_table(:,ind));
gp_var = gp_var_cell{maxind};
%%% Extract inducing points and signal_var from VAR %%%
xu=gp_var.X_u;
signal_var=gp_var.lik.sigma2;

%%% Compute UB to NLD %%%
K_mn=gp_cov(gp_var,xu,x); K_mm=gp_trcov(gp_var,xu);
L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
L=L_mm'\K_mn; %L'*L=K_hat=K_mn'*(K_mm\K_mn)
A=L*L'+signal_var*eye(m);
L_naive=chol(A);
naive_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_naive)));
K_naive=L'*L;

[K,C]=gp_trcov(gp_var,x);
%%% Compute true nld, nip and hence ml with hyp in var_gp %%%
Ltemp = chol(C); ztemp = Ltemp'\y ;
nld(ind) = -sum(log(diag(Ltemp)));
nip(ind) = -sum(ztemp.^2)/2;

%%% Compute UB to NIP %%%
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
cg_obj_table(ind)=cg_obj(end);
pcg_obj_table(ind)=pcg_obj(end);
pcg_objf_table(ind)=pcg_objf(end);
pcg_objp_table(ind)=pcg_objp(end);
naive_nld_table(ind)=naive_nld;

%%% Compute RFF and use as UB to NLD %%%
l1=gp_var.cf{1}.cf{2}.cf{1}.lengthScale; sf1=gp_var.cf{1}.cf{2}.cf{1}.magnSigma2;
l2=gp_var.cf{1}.cf{2}.cf{2}.lengthScale; sf2=gp_var.cf{1}.cf{2}.cf{2}.magnSigma2;
lper=gp_var.cf{1}.cf{1}.lengthScale; sfper=gp_var.cf{1}.cf{1}.magnSigma2; per=gp_var.cf{1}.cf{1}.period;
parfor i=1:10
rng(i);
phi=solarRFF(x,m,l1,sf1,l2,sf2,lper,per,sfper);
Ar=phi*phi'+signal_var*eye(m);
L_rff=chol(Ar);
rff_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_rff)));
rff_nld_table(i,ind)=rff_nld;
fprintf('RFF for worker %d done \n',i);
end
ind=ind+1;
end
ml = nld + nip - n*log(2*pi)/2;
end
delete(POOL)