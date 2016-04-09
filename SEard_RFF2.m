function phi=SEard_RFF2(x,length_scale,sigma_RBF,Z)
    %function to generate RFF for SEard kernel
    %Z~randn(half_m,D);
    [~,D]=size(x);
    half_m=size(Z,1);
    length_scale=reshape(length_scale,[1,D]);
    for i=1:half_m
        Z(i,:)=Z(i,:)./length_scale;
    end
    temp=Z*x';
    sin_feat=sin(temp); cos_feat=cos(temp);
    phi=reshape([sin_feat(:) cos_feat(:)]',2*half_m, []);
    phi=sigma_RBF*phi/sqrt(half_m);
end