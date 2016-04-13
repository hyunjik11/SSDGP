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
[K,~]=gp_trcov(gp,x);

cg_ub=zeros(6,1); pcg_ub=zeros(6,1);
k=1;
for m=[10,20,40,80,160,320]
    [ub,~,cg_resvec]=ip_ub(K,y,signal_var,m);
    cg_ub(k)=ub;
    idx=randsample(n,m);
    K_mn=K(idx,:); K_mm=K(idx,idx);
    L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
    L=L_mm'\K_mn; %L'*L=K_hat=K_mn'*(K_mm\K_mn)
    K_naive=L'*L;
    K_pic=K_naive+blockdiag(K-K_naive,m);
    [ub,~,pcg_resvec]=ip_ub(K,y,signal_var,m,K_pic);
    pcg_ub(k)=ub;
    k=k+1;
    fprintf('m=%d done',m);
end
figure();
subplot(1,2,1);
plot(cg_resvec);hold on; plot(pcg_resvec);
legend('CG','PCG')
xlabel('iter')
ylabel('residual Ax-b')
ylim([0 100])
hold off
subplot(1,2,2);
plot(-half_innerprod*ones(6,1)); hold on;
plot(cg_ub); plot(pcg_ub);
legend('exact -ip/2','CG','PCG')
set(gca,'XTick',[1 2 3 4 5 6]);
set(gca,'XTickLabel',[10 20 40 80 160 320]);
xlabel('iter')
ylabel('upper bound on -ip/2')
