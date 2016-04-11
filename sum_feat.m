function phi=sum_feat(phi1,phi2,idx)
%function to generate random features for the sum of kernels k1,k2
%with random features phi1,phi2, both of size m by n
%idx=indices of the m chosen features out of 2m
%e.g. idx=randsample(2*m,m);
temp=[phi1;phi2];
phi=sqrt(2)*temp(idx,:);
end
