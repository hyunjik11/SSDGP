function [phi,z]=RFFsum(x,z1,z2,s1,s2)
    %function to generate RFF for sum of kernels k1,k2
    %z1,z2 are the sampled spectral frequencies from spectral densities of k1,k2
    %z1,z2 both half_m by D
    %wish to generate z, the spectral frequencies for the sum, then phi.
    %s1,s2 are sigmaRBF2 of k1,k2
    %use RFF2 by default
    [half_m,D]=size(z1);
    if size(z1)~=size(z2);
        error('z1 and z2 have different dimensions')
    end
    unif=rand(half_m,1);
    idx=(unif<s1/(s1+s2));
    z=zeros(half_m,D);
    for i=1:half_m
        if idx(i)
            z(i,:)=z1(i,:);
        else z(i,:)=z2(i,:);
        end
    end
    %z=z1.*idx+z2.*(~idx); %draws from mixture of the two spectral distribs
    phi=RFF2(x,z,sqrt(s1+s2));
end