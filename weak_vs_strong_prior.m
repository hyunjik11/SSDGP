load solar.mat
x=X; Y=y;
x_mean=mean(x); x_std=std(x);
y_mean=mean(y); y_std=std(y);
x = (x-x_mean)/x_std; %normalise x;
y = (y-y_mean)/y_std; %normalise y;
[n,D]=size(x);

opt = optimset('TolFun',1e-6,'TolX',1e-6,'Display','iter','MaxIter',1000);

lik=lik_init(y);
gpcf_se_r=se_init(x,y);
gp_r = gp_set('lik',lik,'cf',gpcf_se_r);
gp_r = gp_optim(gp_r,x,y,'opt',opt,'optimf',@fminscg);

rng(120) 
gpcf_se_wp = se_init(x,y); gpcf_se_wp.p.lengthScale=prior_unif();
lik = lik_init(y); lik.p.sigma2 = prior_unif();
gp_wp = gp_set('lik',lik,'cf',gpcf_se_wp);
gp_wp = gp_optim(gp_wp,x,y,'opt',opt,'optimf',@fminscg);

 
gpcf_se_wi = se_init(x,y); gpcf_se_wi.lengthScale=0.0001;
lik = lik_init(y); lik.sigma2 = 0.0001;
gp_wi = gp_set('lik',lik,'cf',gpcf_se_wi);
gp_wi = gp_optim(gp_wi,x,y,'opt',opt,'optimf',@fminscg);

gpcf_se_w = gpcf_sexp('lengthScale',0.00001,'lengthScale_prior',prior_unif());
lik = lik_gaussian('sigma2',0.0001,'sigma2_prior',prior_unif());
gp_w = gp_set('lik',lik,'cf',gpcf_se_w);
gp_w = gp_optim(gp_w,x,y,'opt',opt,'optimf',@fminscg);

pred_r=gp_pred(gp_r,x,y,x);
pred_r=pred_r*y_std+y_mean;

pred_wp=gp_pred(gp_wp,x,y,x);
pred_wp=pred_wp*y_std+y_mean;

pred_wi=gp_pred(gp_wi,x,y,x);
pred_wi=pred_wi*y_std+y_mean;

pred_w=gp_pred(gp_w,x,y,x);
pred_w=pred_w*y_std+y_mean;

scatter(X,Y,'x');
hold on
plot(X,pred_r,'lineWidth',2);
plot(X,pred_wp,'lineWidth',2.5);
plot(X,pred_wi,'lineWidth',2);
plot(X,pred_w,'lineWidth',1.5);
hold off
legend('data','strong prior, rand init','weak prior, rand init','strong prior, small init','weak prior, small init')
