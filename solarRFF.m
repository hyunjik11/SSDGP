function phi=solarRFF(x,m,l1,sf1,l2,sf2,lper,per,sfper)
%function to generate random features for solar kernel (SE+SE)PER
%m is dimenionality of phi
half_m=m/2;
z1=randSE(l1,half_m);
z2=randSE(l2,half_m);
[~,z]=RFFsum(x,z1,z2,sf1,sf2);
z3=randPER(per,lper,half_m);

phi=RFFprod(x,z,z3,sqrt(sf1*sf2),sfper);
end