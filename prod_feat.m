function phi=prod_feat(phi1,phi2,idx)
%function to generate random features for the product of kernels k1,k2
%with random features phi1,phi2, both of size m by n
%idx=indices of the m chosen features out of m^2 in kron
%e.g. idx=randsample(m^2,m);
[m,n]=size(phi1);
phi=zeros(m,n);
for k=1:m
    i=rem(idx(k),m)+1;
    j=(idx(k)-i+1)/m +1;
    phi(k,:)=sqrt(m)*phi1(i,:).*phi2(j,:);
end
end
