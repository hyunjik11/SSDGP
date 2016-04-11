function x=discretesample(y,p,m)
    %get m samples from discrete distrib where Pr(X=y(i))=p(i)
    idx=discretize(rand(1,m),[0;cumsum(p(:))/sum(p)]);
    x=y(idx);
end
