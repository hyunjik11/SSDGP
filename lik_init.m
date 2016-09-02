function lik=lik_init(y)
    std_y = std(y);
    lik = lik_gaussian('sigma2',0.1*std_y*exp(randn()/2));
end