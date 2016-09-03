function gpcf=se_init(x,y)
    std_x = std(x);
    std_y = std(y);
    pl=prior_gaussian('s2',0.25);
    gpcf = gpcf_sexp('lengthScale',std_x*exp(randn()/2), 'magnSigma2',...
        0.1*std_y*exp(randn()/2),'lengthScale_prior',pl);
end
