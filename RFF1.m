function phi=RFF1(x,z,b,sigma_RBF)
    %function to generate 1D RFF of first type using frequencies z sampled
    %from spectral density.
    %z is m by D; b is m by 1; x is n by D
    [n,D]=size(x);
    m=size(z,1);
    phi=z*x';
    for i=1:n
        phi(:,i)=cos(phi(:,i)+b);
    end
    phi=sqrt(2/m)*sigma_RBF*phi;
end
    