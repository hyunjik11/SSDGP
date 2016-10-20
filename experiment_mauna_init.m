addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));
parallel=1; subset=1; half=1; learn_ind_pts = 1;
if subset
    fprintf('Using subset of training init ');
else
    fprintf('Using Kmeans init ');
end
if half
    fprintf('with half rand, half connected init \n');
else
    fprintf('with all connected init \n');
end
if parallel
    num_workers=10;
    POOL=parpool('local',num_workers);
end
warning('off','all');
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
ne = zeros(1,lm);
ne_full = zeros(numiter,1);
gp_var_cell=cell(1,numiter);
idx_u_cell=cell(1,numiter);

for m=m_values
fprintf('m=%d \n',m);
parfor i=1:numiter
rng(i)
warning('off','all');
%t=getCurrentTask(); k=t.ID;
%filename=['experiment_results/mauna10m',num2str(k),'.txt'];
%fileID=fopen(filename,'at');

%%% Initialise gp_var %%%
if ind == 1 || mod(half*i+half+1,2)==0 %latter = 1 if half=0, and mod(i,2) if half=1
    lik=lik_init(y);
    gpcf_se1=se_init(x,y);
    gpcf_lin=lin_init();
    gpcf_se2=se_init(x,y);
    gpcf_per=per_init(x,y);
    gpcf1=gpcf_prod('cf',{gpcf_se1,gpcf_lin});
    gpcf2=gpcf_prod('cf',{gpcf_se2,gpcf_per});
    gpcf3=se_init(x,y);
    if subset
        [X_u,idx]=datasample(x,m,1,'Replace',false); %random initialisation
        idx_u_cell{i}=idx; %store indices of subset
    else
        [~,X_u]=kmeans(x,m); %inducing pts initialised by Kmeans++
    end
    gp_var_loc = gp_set('type', 'VAR', 'lik', lik, 'cf',{gpcf1,gpcf2,gpcf3},'X_u', X_u); 
    if learn_ind_pts
        gp_var_loc = gp_set(gp_var_loc, 'infer_params', 'covariance+likelihood+inducing');
    else
        gp_var_loc = gp_set(gp_var_loc, 'infer_params', 'covariance+likelihood');
    end
else
    if subset %select m_new-m_old more pts from training, with no overlap
        weights = 1e-10*ones(1,size(x,1)); %weights for sampling
        weights(idx_u)=1; %make sure samples idx_u are included
        [X_u,idx] = datasample(x,m,1,'Replace',false,'Weights',weights);
        idx_u_cell{i}=idx; %store indices of subset
    else %add m_new-m_old more pts to X_u, initialised by kmeans
        [~,X_u_new] = kmeans(x,m-m_old);
        X_u = [xu;X_u_new];
    end
    gp_var_loc = gp_var;
    gp_var_loc.X_u = X_u; %keep hyperparams & ind pts the same, but add more ind pts.
end

opt=optimset('TolFun',1e-4,'TolX',1e-5,'Display','off','MaxIter',1000);

%%% Optimise gp %%%
if ind == 1
    gp = gp_set('lik',lik,'cf',{gpcf1,gpcf2,gpcf3});
    gp = gp_optim(gp,x,y,'opt',opt,'optimf',@fminscg);
    energy = gp_e([],gp,x,y);
    ne_full(i) = -energy;
end

%%% Optimise gp_var %%%
gp_var_loc=gp_optim(gp_var_loc,x,y,'opt',opt,'optimf',@fminscg);
energy=gp_e([],gp_var_loc,x,y);

%%% Record ml and gp_var %%%
lb_table(i,ind)=-energy;
gp_var_cell{i} = gp_var_loc;
%fprintf('optim for worker %d done \n',i);
end

%%% Get index of best var LB %%%
[~,maxind]=max(lb_table(:,ind));
gp_var = gp_var_cell{maxind};
if subset
    idx_u = idx_u_cell{maxind};
end

%%% Compute ne of full GP for these hyp %%%
gp = gp_set('lik',gp_var.lik,'cf',gp_var.cf);
energy = gp_e([],gp,x,y);
ne(ind) = -energy;

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

parfor j=1:numiter
idx1=randsample(m^2,m);
idx2=randsample(2*m,m);
idx3=randsample(2*m,m);
phi=maunaRFF(x,m,l1,sf1,cs2,l2,sf2,lper,per,sfper,l3,sf3,idx1,idx2,idx3);
Ar=phi*phi'+signal_var*eye(m);
L_rff=chol(Ar);
rff_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_rff)));
rff_nld_table(j,ind)=rff_nld;
%fprintf('RFF for worker %d done \n',i);
end
% fprintf('ml=%4.3f \n',nld(ind)+nip(ind)-n*log(2*pi)/2);
% fprintf('SE1 magnSigma2=%4.3f, l=%4.3f \n',...
%     gp_var.cf{1}.cf{1}.magnSigma2, gp_var.cf{1}.cf{1}.lengthScale);
% fprintf('LIN coeffSigma2=%4.3f \n',...
%     gp_var.cf{1}.cf{2}.coeffSigma2);
% fprintf('SE2 magnSigma2=%4.3f, l=%4.3f \n',...
%     gp_var.cf{2}.cf{1}.magnSigma2, gp_var.cf{2}.cf{1}.lengthScale);
% fprintf('PER magnSigma2=%4.3f, l=%4.3f, per=%4.3f \n',...
%     gp_var.cf{2}.cf{2}.magnSigma2, gp_var.cf{2}.cf{2}.lengthScale, gp_var.cf{2}.cf{2}.period);
% fprintf('SE3 magnSigma2=%4.3f, l=%4.3f \n',...
%     gp_var.cf{3}.magnSigma2, gp_var.cf{3}.lengthScale);
% fprintf('lik sigma2=%4.8f \n',gp_var.lik.sigma2);

ind=ind+1;
m_old=m;
end
delete(POOL)