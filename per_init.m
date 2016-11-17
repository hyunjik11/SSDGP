function gpcf=per_init(x,y,dim)
    % optional: dim is the dimension on which the se kernel is defined
    if size(x,2) > 1
        x = x(:,dim);
    end
    std_x = std(x);
    std_y = std(y);
    n=length(x);
    pl=prior_gaussian('mu',std_x,'s2',0.25);
    minp = log(10*(max(x)-min(x))/n);
    plper=prior_gaussian('mu',minp+1,'s2',0.25);
    if nargin == 2
        gpcf = gpcf_periodic('lengthScale',std_x*exp(randn()/2),...
        'period',exp(minp+TruncatedGaussian(-1/2,[0,Inf])),'magnSigma2',...
        0.1*std_y*exp(randn()/2),'lengthScale_prior',prior_t(),...
        'period_prior',plper);
    else
        gpcf = gpcf_periodic('lengthScale',std_x*exp(randn()/2),...
        'period',exp(minp+TruncatedGaussian(-1/2,[0,Inf])),'magnSigma2',...
        0.1*std_y*exp(randn()/2),'lengthScale_prior',prior_t(),...
        'period_prior',plper,'selectedVariables',dim);
    end
end