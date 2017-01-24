function draw_gp_lin(coeffSigma2)
    lik = lik_gaussian('sigma2',0.001);
    gpcf = gpcf_linear('coeffSigma2',coeffSigma2);
    gp=gp_set('lik',lik,'cf',gpcf);
    x = linspace(-1,1)';
    K = gp_trcov(gp,x);
    L = chol(K)'*randn(100,1);
    plot(x,L)
    xlim([-1,1])
end