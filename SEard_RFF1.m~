function phi=SEard_RFF1(x,length_scale,sigma_RBF,Z,b)
    %function to generate RFF for SEard kernel
    %Z~randn(m,D);b~2*pi*rand(m);
    [n,D]=size(x);
    m=size(Z,1);
    length_scale=reshape(length_scale,[1,D]);
    b=reshape(b,[m,1]);
    for i=1:m
        Z(i,:)=Z(i,:)./length_scale;
    end
    phi=Z*x';
    for i=1:n
        phi(:,i)=cos(phi(:,i)+b);
    end
    phi=sqrt(2/m)*sigma_RBF*phi;
end