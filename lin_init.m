function gpcf=lin_init(dim)
    % optional: dim is the dimension on which the se kernel is defined
    if nargin == 0
        gpcf = gpcf_linear('coeffSigma2',exp(randn()/2));
    else
        gpcf = gpcf_linear('selectedVariables',dim, ...
            'coeffSigma2',exp(randn()/2));
end