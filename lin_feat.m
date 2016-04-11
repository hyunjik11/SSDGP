function phi=lin_feat(x,m,coeffSigma2)
    %compute linear features sqrt(coeffSigma2)*x'
    %and make m copies
    phi=sqrt(coeffSigma2/m)*repmat(x',m,1);
end
