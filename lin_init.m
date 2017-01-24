function gpcf=lin_init(dim)
    % optional: dim is the dimension on which the se kernel is defined
    if nargin == 0
        gpcf = gpcf_linear_loc('coeffSigma2',TruncatedGaussian(-1/2,[0,Inf]),'loc',randn()/2);
    else
        gpcf = gpcf_linear_loc('selectedVariables',dim, ...
            'coeffSigma2',TruncatedGaussian(-1/2,[0,Inf]),'loc',randn()/2);
    end
end