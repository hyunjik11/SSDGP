function lik=lik_init(y)
    std_y = std(y);
    lik = lik_gaussian('sigma2',0.1*std_y*exp(randn()/2),...
        'sigma2_prior',prior_gaussian('mu',0,'s2',0.2));
end