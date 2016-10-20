addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
time=1; subset=1;
if subset
    fprintf('Using subset of training with rand init \n')
else
    fprintf('Using Kmeans with rand init \n')
end
num_workers=10;
%POOL=parpool('local',num_workers);

%%% load and set up data %%%
load mauna.txt
z = mauna(:,2) ~= -99.99; % get rid of missing data
x = mauna(z,1); y = mauna(z,2); % extract year and CO2 concentration
X = x; Y = y;
x_mean=mean(x); x_std=std(x);
y_mean=mean(y); y_std=std(y);
x = (x-x_mean)/x_std; %normalise x;
y = (y-y_mean)/y_std; %normalise y;
[n,D]=size(x);

%%%%%%%%%%%%%%%%%%%
numiter=10;
m_values=[10,20,40,80,160,320];
lm=length(m_values);

if time
gp_time = zeros(numiter,1);
var_time = zeros(numiter,lm);
nld_time = zeros(numiter,lm);
nip_time = zeros(numiter,lm);
parfor iter = 1:numiter
    warning('off','all');
for ind=1:lm
    m = m_values(ind);

%%% Initialise gp_var %%%
lik=lik_init(y);
gpcf_se1=se_init(x,y);
gpcf_lin=lin_init();
gpcf_se2=se_init(x,y);
gpcf_per=per_init(x,y);
gpcf1=gpcf_prod('cf',{gpcf_se1,gpcf_lin});
gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
gpcf3=se_init(x,y);

opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);

%%% Optimise gp %%%
if m==min(m_values)
    gp = gp_set('lik',lik,'cf',{gpcf1,gpcf2,gpcf3});
    tic;
    gp = gp_optim(gp,x,y,'opt',opt,'optimf',@fminscg);
    time = toc;
    gp_time(iter)=time;
end

%%% Optimise gp_var %%%
if subset
    X_u = datasample(x,m,1,'Replace',false); 
else
    [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
end
gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3},'X_u', X_u); 
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
tic;
gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
time=toc;
var_time(iter,ind) = time;
%fprintf('Time taken for optimising VAR: %4.2f s \n',time);
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
%fprintf('Time taken for computing L_naive: %4.4f s \n',time);
nld_time(iter,ind) = time;
naive_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_naive)));
K_naive=L'*L;

%%% Compute RFF and use as UB to NLD %%%
% l1=gp_var.cf{1}.cf{1}.lengthScale; sf1=gp_var.cf{1}.cf{1}.magnSigma2;
% cs2=gp_var.cf{1}.cf{2}.coeffSigma2;
% l2=gp_var.cf{2}.cf{1}.lengthScale; sf2=gp_var.cf{2}.cf{1}.magnSigma2;
% lper=gp_var.cf{2}.cf{2}.lengthScale; sfper=gp_var.cf{2}.cf{2}.magnSigma2; per=gp_var.cf{2}.cf{2}.period;
% l3=gp_var.cf{3}.lengthScale; sf3=gp_var.cf{3}.magnSigma2;
% tic;
% idx1=randsample(m^2,m);
% idx2=randsample(2*m,m);
% idx3=randsample(2*m,m);
% phi=maunaRFF(x,m,l1,sf1,cs2,l2,sf2,lper,per,sfper,l3,sf3,idx1,idx2,idx3);
% Ar=phi*phi'+signal_var*eye(m);
% L_rff=chol(Ar);
% time=toc;
% %fprintf('Time taken for computing L_rff: %4.4f s \n',time);
% rff_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_rff)));

%%% Compute UB to NIP %%%
[K,C]=gp_trcov(gp_var,x);
tic;
[~,~,~,~,cg_resvec,cg_obj]=cgs_obj(C,y,[],m);
time=toc;
%fprintf('Time taken for CG: %4.4f s \n',time);
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

tic; [~,~,~,~,pcg_resvec,pcg_obj]=cgs_obj(C,y,[],m,myfun); time=toc;
%fprintf('Time taken for PCG Nystrom: %4.4f s \n',time);
tic; [~,~,~,~,pcg_resvecf,pcg_objf]=cgs_obj(C,y,[],m,myfunf); time=toc;
%fprintf('Time taken for PCG FIC: %4.4f s \n',time);
tic;
Ap=L*invfun(L')+eye(m);
myfunp = @(w) invfun(w-L'*(Ap\(L*invfun(w))));
[~,~,~,~,pcg_resvecp,pcg_objp]=cgs_obj(C,y,[],m,myfunp); time=toc;
nip_time(iter,ind) = time;
%fprintf('Time taken for PCG PIC: %4.4f s \n',time);
cg_obj=minsofar(cg_obj);
pcg_obj=minsofar(pcg_obj);
pcg_objf=minsofar(pcg_objf);
pcg_objp=minsofar(pcg_objp);

%fprintf('worker %d done \n',i);
%end

end
end

else
lb_table=zeros(numiter,lm);
naive_nld_table=zeros(1,lm);
rff_nld_table=zeros(numiter,lm);
cg_obj_table=zeros(1,lm);
pcg_obj_table=zeros(1,lm);
pcg_objf_table=zeros(1,lm);
pcg_objp_table=zeros(1,lm);
nld = zeros(1,lm);
nip = zeros(1,lm);
gp_var_cell=cell(1,numiter);

for m=m_values
fprintf('m=%d \n',m);
parfor i=1:numiter
rng(i+100)
warning('off','all');
%t=getCurrentTask(); k=t.ID;
%filename=['experiment_results/mauna10m',num2str(k),'.txt'];
%fileID=fopen(filename,'at');


%%% Initialise gp_var %%%
lik=lik_init(y);
gpcf_se1=se_init(x,y);
gpcf_lin=lin_init();
gpcf_se2=se_init(x,y);
gpcf_per=per_init(x,y);
gpcf1=gpcf_prod('cf',{gpcf_se1,gpcf_lin});
gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
gpcf3=se_init(x,y);

%%% Optimise gp_var %%%
if subset
    X_u = datasample(x,m,1,'Replace',false); 
else
    [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
end
gp_var = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3},'X_u', X_u); 
gp_var = gp_set(gp_var, 'infer_params', 'covariance+likelihood');
opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);
gp_var=gp_optim(gp_var,x,y,'opt',opt,'optimf',@fminscg);
[~,nll]=gp_e([],gp_var,x,y);

%%% Record ml and gp_var %%%
lb_table(i,ind)=-nll;
gp_var_cell{i} = gp_var;
%fprintf('optim for worker %d done \n',i);
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
naive_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_naive))); %the first term comes from matrix identity.
% so this is an upper bound on -logdet(K+signal_var*I)
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

% Gather all results
cg_obj_table(ind)=cg_obj(end);
pcg_obj_table(ind)=pcg_obj(end);
pcg_objf_table(ind)=pcg_objf(end);
pcg_objp_table(ind)=pcg_objp(end);
naive_nld_table(ind)=naive_nld;

%%% Compute RFF and use as UB to NLD %%%
l1=gp_var.cf{1}.cf{1}.lengthScale; sf1=gp_var.cf{1}.cf{1}.magnSigma2;
cs2=gp_var.cf{1}.cf{2}.coeffSigma2;
l2=gp_var.cf{2}.cf{1}.lengthScale; sf2=gp_var.cf{2}.cf{1}.magnSigma2;
lper=gp_var.cf{2}.cf{2}.lengthScale; sfper=gp_var.cf{2}.cf{2}.magnSigma2; per=gp_var.cf{2}.cf{2}.period;
l3=gp_var.cf{3}.lengthScale; sf3=gp_var.cf{3}.magnSigma2;

parfor i=1:numiter
idx1=randsample(m^2,m);
idx2=randsample(2*m,m);
idx3=randsample(2*m,m);
phi=maunaRFF(x,m,l1,sf1,cs2,l2,sf2,lper,per,sfper,l3,sf3,idx1,idx2,idx3);
Ar=phi*phi'+signal_var*eye(m);
L_rff=chol(Ar);
rff_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_rff)));
rff_nld_table(i,ind)=rff_nld;
%fprintf('RFF for worker %d done \n',i);
end
    fprintf('ml=%4.3f \n',nld(ind)+nip(ind)-n*log(2*pi)/2);
    fprintf('SE1 magnSigma2=%4.3f, l=%4.3f \n',...
        gp_var.cf{1}.cf{1}.magnSigma2, gp_var.cf{1}.cf{1}.lengthScale);
    fprintf('LIN coeffSigma2=%4.3f \n',...
        gp_var.cf{1}.cf{2}.coeffSigma2);
    fprintf('SE2 magnSigma2=%4.3f, l=%4.3f \n',...
        gp_var.cf{2}.cf{1}.magnSigma2, gp_var.cf{2}.cf{1}.lengthScale);
    fprintf('PER magnSigma2=%4.3f, l=%4.3f, per=%4.3f \n',...
        gp_var.cf{2}.cf{2}.magnSigma2, gp_var.cf{2}.cf{2}.lengthScale, gp_var.cf{2}.cf{2}.period);
    fprintf('SE3 magnSigma2=%4.3f, l=%4.3f \n',...
        gp_var.cf{3}.magnSigma2, gp_var.cf{3}.lengthScale);
    fprintf('lik sigma2=%4.8f \n',gp_var.lik.sigma2);

ind=ind+1;
end
ml = nld + nip - n*log(2*pi)/2;
delete(POOL)
end




