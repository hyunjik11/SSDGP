function phi=RFF1(x,z,b,sigma_RBF)
    %function to generate 1D RFF of first type using frequencies z sampled
    %from spectral density.
    %z~randn(m,1); b~2*pi*rand(m,1); NOTE both are column vectors. So is x
    n=length(x);
    m=length(z);
    phi=z*x';
    for i=1:n
        phi(:,i)=cos(phi(:,i)+b);
    end
    phi=sqrt(2/m)*sigma_RBF*phi;
end
    