function phi=RFFprod(x,z1,z2,s1,s2)
    %function to generate RFF for product of kernels k1,k2
    %z1,z2 are the sampled spectral frequencies from spectral densities of k1,k2
    %z1,z2 both half_m by D
    %wish to generate z, the spectral frequencies for the product, then phi.
    %s1,s2 are sigmaRBF2 of k1,k2
    %use RFF2 by default
    z=z1+z2; %draws from mixture of the two spectral distribs
    phi=RFF2(x,z,sqrt(s1*s2));
end