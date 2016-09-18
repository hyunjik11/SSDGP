grad = gpcf_per.fh.lpg(gpcf_per);
grad = grad(2);
fprintf('diff_grad = ')
for eps = [1e-2,1e-3,1e-4,1e-5]
    gpcf_temp=gpcf_per; 
    gpcf_temp.lengthScale = exp(log(gpcf_per.lengthScale) + eps);
    val_new = gpcf_temp.fh.lp(gpcf_temp);
    gpcf_temp=gpcf_per; 
    gpcf_temp.lengthScale = exp(log(gpcf_per.lengthScale) - eps);
    val_old = gpcf_temp.fh.lp(gpcf_temp);
    fd = (val_new - val_old)/(2*eps);
    fprintf('%4.8f ',abs((fd-grad)/grad))
end
fprintf('\n')

grad = gpcf_per.fh.cfg(gpcf_per,x,X_u);
grad = grad{2};
fprintf('diff_grad = ')
for eps = [1e-2,1e-3,1e-4,1e-5]
    gpcf_temp=gpcf_per; 
    gpcf_temp.lengthScale = exp(log(gpcf_per.lengthScale) + eps);
    val_new = gpcf_temp.fh.cov(gpcf_temp,x,X_u);
    gpcf_temp=gpcf_per; 
    gpcf_temp.lengthScale = exp(log(gpcf_per.lengthScale) - eps);
    val_old = gpcf_temp.fh.cov(gpcf_temp,x,X_u);
    fd = (val_new - val_old)/(2*eps);
    fprintf('%4.8f ',sum(sum(abs(fd-grad))))
end
fprintf('\n')