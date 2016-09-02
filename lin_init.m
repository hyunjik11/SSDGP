function gpcf=lin_init()
    gpcf = gpcf_linear('coeffSigma2',TruncatedGaussian(-1,[1,Inf]));
end