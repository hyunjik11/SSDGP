function phi=solarRFF(x,m,l1,sf1,l2,sf2,lper,per,sfper)
%function to generate random features for solar kernel (SE+SE)PER
%m is dimenionality of phi
half_m=m/2;
z1=randSE(l1,half_m);
z2=randSE(l2,half_m);
unif=rand(half_m,1);
idx=(unif<sf1/(sf1+sf2));
z=z1.*idx+z2.*(~idx);
z3=randPER(per,lper,half_m);

phi=RFFprod(x,z,z3,sqrt(sf1*sf2),sfper);
end