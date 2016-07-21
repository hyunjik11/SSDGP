function phi=maunaRFF(x,m,l1,sf1,cs2,l2,sf2,lper,per,sfper,l3,sf3,idx1,idx2,idx3)
%function to generate random features for mauna kernel
%m is the dimensionality of phi
%idx1=randsample(m^2,m)
%idx2,idx3=randsample(2*m,m)
%rest are hyperparam values
z1=randSE(l1,m);
b1=2*pi*rand(m,1);
phi_se1=RFF1(x,z1,b1,sqrt(sf1));
phi_lin=lin_feat(x,m,cs2);
phi1=prod_feat(phi_se1,phi_lin,idx1);

half_m=m/2;
z2=randSE(l2,half_m);
z3=randPER(per,lper,half_m);
phi2=RFFprod(x,z2,z3,sf2,sfper);
 
%z4=randSE(l3,half_m);
%z5=randRQ(alpha,lrq,half_m);
%phi3=RFFprod(x,z4,z5,sf3,sfrq);
phi3=SEard_RFF2(x,l3,sqrt(sf3),randn(half_m,1));

phi_temp=sum_feat(phi1,phi2,idx2);
phi=sum_feat(phi_temp,phi3,idx3);
end