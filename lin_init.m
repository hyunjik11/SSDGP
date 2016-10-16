function gpcf=lin_init(dim)
    % optional: dim is the dimension on which the se kernel is defined
    if nargin == 0
        gpcf = gpcf_linear('coeffSigma2',TruncatedGaussian(-1/2,[0,Inf]));
    else
        gpcf = gpcf_linear('selectedVariables',dim, ...
            'coeffSigma2',TruncatedGaussian(-1/2,[0,Inf]));
    end
end