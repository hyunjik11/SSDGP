function gpcf=se_init(x,y,dim)
    % optional: dim is the dimension on which the se kernel is defined
    std_x = std(x);
    std_y = std(y);
    pl=prior_gaussian('s2',0.25);
    if nargin == 2
        gpcf = gpcf_sexp('lengthScale',std_x*exp(randn()/2), 'magnSigma2',...
        0.1*std_y*exp(randn()/2),'lengthScale_prior',pl);
    else 
        gpcf = gpcf_sexp('selectedVariables',dim,'lengthScale',...
            std_x*exp(randn()/2), 'magnSigma2',...
        0.1*std_y*exp(randn()/2),'lengthScale_prior',pl);
end
