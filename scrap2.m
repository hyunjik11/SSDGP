addpath(genpath('/homes/hkim/Documents/GPstuff-4.6')); 
load solar.mat
num_workers=10;
%POOL=parpool('local',num_workers);

meanX=mean(X); meany=mean(y);
stdX=std(X); stdy=std(y);

yw=(y-meany)/stdy;

[n, D] = size(X);
per=1; %periodicity of data


lik=lik_gaussian();
gpcf_se1=gpcf_sexp();
gpcf_se2=gpcf_sexp();
gpcf_per = gpcf_periodic('period',per,'period_prior',prior_logunif(),'lengthScale_sexp_prior',prior_t());
gpcf_se=gpcf_sum('cf',{gpcf_se1,gpcf_se2});
%gpcf=gpcf_prod('cf',{gpcf_per,gpcf_se});

gp=gp_set('lik',lik,'cf',gpcf_se);
opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','iter','MaxIter',1000);
gp=gp_optim(gp,X,yw,'opt',opt);

gpcf=gpcf_prod('cf',{gp.cf{1},gpcf_per});

gp=gp_set('lik',gp.lik,'cf',gpcf);
opt=optimset('TolFun',1e-3,'TolX',1e-4,'Display','iter','MaxIter',1000);
gp=gp_optim(gp,X,yw,'opt',opt);


pred=gp_pred(gp,X,yw,X);
pred=pred*stdy+meany;
scatter(X,y);
hold on
plot(X,pred);

%delete(POOL)
