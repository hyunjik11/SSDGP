function phi=RFF2(x,z,sigma_RBF)
    %function to generate 1D RFF of second type using frequencies z sampled
    %from spectral density.
    %z~randn(half_m,1); NOTE z is a column vector. So is x
    n=length(x);
    half_m=length(z);
    temp=z*x';
    sin_feat=sin(temp); cos_feat=cos(temp);
    phi=reshape([sin_feat(:) cos_feat(:)]',2*half_m, []);
    phi=sigma_RBF*phi/sqrt(half_m);
end