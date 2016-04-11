function phi=RFFsum(x,z1,z2,s1,s2)
    %function to generate RFF for sum of kernels k1,k2
    %z1,z2 are the sampled spectral frequencies from spectral densities of k1,k2
    %wish to generate z, the spectral frequencies for the sum, then phi.
    %want length(z)=length(z1)=length(z2);
    %s1,s2 are sigmaRBF2 of k1,k2
    %use RFF2 by default
    half_m=length(z1);
    unif=rand(half_m,1);
    idx=(unif<s1/(s1+s2));
    z=z1.*idx+z2.*(~idx); %draws from mixture of the two spectral distribs
    phi=RFF2(x,z,sqrt(s1+s2));
end