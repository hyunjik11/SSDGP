%plot residual against iter for CG and PCG using PIC as preconditioner
%also plot upper bound against iteration for m=10,20,40,80,160,320
%addpath(genpath('/homes/hkim/Documents/GPstuff-4.6'));
addpath(genpath('/Users/hyunjik11/Documents/GPstuff'));

x=h5read('PPdata_full.h5','/Xtrain');
y=h5read('PPdata_full.h5','/ytrain');

length_scale=[1.3978 0.0028 2.8966 7.5565];
sigma_RBF2=0.8333; 
signal_var=0.0195;
[n, D] = size(x);
half_innerprod=4784.0;

lik = lik_gaussian('sigma2', signal_var);
gpcf = gpcf_sexp('lengthScale', length_scale, 'magnSigma2', sigma_RBF2);
gp=gp_set('lik',lik,'cf',gpcf); %exact gp
[K,C]=gp_trcov(gp,x);

m=320;
tic;
[~,~,~,~,cg_resvec,cg_obj]=cgs_obj(C,y,[],m);
time=toc;
fprintf('Time taken for CG: %4.2f s \n',time);
idx=randsample(n,m);
K_mn=K(idx,:); K_mm=K(idx,idx);
L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
L=L_mm'\K_mn; %L'*L=K_hat=K_mn'*(K_mm\K_mn)
K_naive=L'*L;
A=L*L'+signal_var*eye(m);
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

tic; [~,~,~,~,pcg_resvec,pcg_obj]=cgs_obj(C,y,[],m,myfun); time=toc;
fprintf('Time taken for PCG Nystrom: %4.2f s \n',time);
tic; [~,~,~,~,pcg_resvecf,pcg_objf]=cgs_obj(C,y,[],m,myfunf); time=toc;
fprintf('Time taken for PCG FIC: %4.2f s \n',time);
tic; [~,~,~,~,pcg_resvecp,pcg_objp]=cgs_obj(C,y,[],m,myfunp); time=toc;
fprintf('Time taken for PCG PIC: %4.2f s \n',time);
cg_obj=minsofar(cg_obj);
pcg_obj=minsofar(pcg_obj);
pcg_objf=minsofar(pcg_objf);
pcg_objp=minsofar(pcg_objp);

figure();
subplot(1,2,1);
plot(cg_resvec);hold on; 
plot(pcg_resvec);
plot(pcg_resvecf);
plot(pcg_resvecp);
legend('CG','PCG-Nystrom','PCG-FIC','PCG-PIC')
xlabel('iter')
ylabel('residual Ax-b')
ylim([0 100])
hold off
subplot(1,2,2);
plot(-half_innerprod*ones(m,1)); hold on;
plot(cg_obj); 
plot(pcg_obj);
plot(pcg_objf);
plot(pcg_objp);
legend('exact -ip/2','CG','PCG-Nystrom','PCG-FIC','PCG-PIC');
xlabel('iter')
ylabel('upper bound on -ip/2')
