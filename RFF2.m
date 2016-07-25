function phi=RFF2(x,z,sigma_RBF)
    %function to generate RFF of second type using frequencies z sampled
    %from spectral density.
    %z is half_m by D, x is n by D (for D=1, both col vectors)
    %phi is m by n
    [n,D]=size(x);
    half_m=size(z,1);
    temp=z*x';
    sin_feat=sin(temp); cos_feat=cos(temp);
    phi=reshape([sin_feat(:) cos_feat(:)]',2*half_m, []);
    phi=sigma_RBF*phi/sqrt(half_m);
end