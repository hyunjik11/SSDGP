function phi=concreteRFF(x,gp,m)
%function to generate random features for concrete kernel
%(wn*lin4)+((se1*se7)+(se1*se2*se4)+(se2*se4*se8))+(se2*se4*se7*se8*lin4)
[n,D]=size(x);
half_m=m/2;
cons1=gp.cf{1}.cf{1}.cf{1}.constSigma2;
cs1_4=gp.cf{1}.cf{2}.coeffSigma2; %dim4
l2_1=gp.cf{2}.cf{1}.lengthScale; sf2_1=gp.cf{2}.cf{1}.magnSigma2; %dim1
l2_7=gp.cf{2}.cf{2}.lengthScale; sf2_7=gp.cf{2}.cf{2}.magnSigma2; %dim7
l3_1=gp.cf{3}.cf{1}.lengthScale; sf3_1=gp.cf{3}.cf{1}.magnSigma2; %dim1
l3_2=gp.cf{3}.cf{2}.lengthScale; sf3_2=gp.cf{3}.cf{2}.magnSigma2; %dim2
l3_4=gp.cf{3}.cf{3}.lengthScale; sf3_4=gp.cf{3}.cf{3}.magnSigma2; %dim4
l4_2=gp.cf{4}.cf{1}.lengthScale; sf4_2=gp.cf{4}.cf{1}.magnSigma2; %dim2
l4_4=gp.cf{4}.cf{2}.lengthScale; sf4_4=gp.cf{4}.cf{2}.magnSigma2; %dim4
l4_8=gp.cf{4}.cf{3}.lengthScale; sf4_8=gp.cf{4}.cf{3}.magnSigma2; %dim8
l5_2=gp.cf{5}.cf{1}.lengthScale; sf5_2=gp.cf{5}.cf{1}.magnSigma2; %dim2
l5_4=gp.cf{5}.cf{2}.lengthScale; sf5_4=gp.cf{5}.cf{2}.magnSigma2; %dim4
l5_7=gp.cf{5}.cf{3}.lengthScale; sf5_7=gp.cf{5}.cf{3}.magnSigma2; %dim7
l5_8=gp.cf{5}.cf{4}.lengthScale; sf5_8=gp.cf{5}.cf{4}.magnSigma2; %dim8
cs5_4=gp.cf{5}.cf{5}.coeffSigma2; %dim4
phi_wn1=wnRFF(n,m,cons1); phi_lin1=lin_feat(x(:,4),m,cs1_4);
idx1=randsample(m^2,m);
phi1=prod_feat(phi_wn1,phi_lin1,idx1);

z2_1=randSE(l2_1,half_m); z2_7=randSE(l2_7,half_m); 
z2=zeros(half_m,D); z2(:,1)=z2_1; z2(:,7)=z2_7; sf2=sf2_1*sf2_7;
z3_1=randSE(l3_1,half_m); z3_2=randSE(l3_2,half_m); z3_4=randSE(l3_4,half_m); 
z3=zeros(half_m,D); z3(:,1)=z3_1; z3(:,2)=z3_2; z3(:,4)=z3_4; sf3=sf3_1*sf3_2*sf3_4;
z4_2=randSE(l4_2,half_m); z4_4=randSE(l4_4,half_m); z4_8=randSE(l4_8,half_m);
z4=zeros(half_m,D); z4(:,2)=z4_2; z4(:,4)=z4_4; z4(:,8)=z4_8; sf4=sf4_2*sf4_4*sf4_8;
[~,ztemp]=RFFsum(x,z2,z3,sf2,sf3);
phi2=RFFsum(x,ztemp,z3,sf2*sf3,sf4);

z5_2=randSE(l5_2,half_m); z5_4=randSE(l5_4,half_m);
z5_7=randSE(l5_7,half_m); z5_8=randSE(l5_8,half_m);
z5=zeros(half_m,D); z5(:,2)=z5_2; z5(:,4)=z5_4; z5(:,7)=z5_7; z5(:,8)=z5_8;
phi_temp=RFF2(x,z5,sqrt(sf5_2*sf5_4*sf5_7*sf5_8));
phi_lin5=lin_feat(x(:,4),m,cs5_4);
idx2=randsample(m^2,m);
phi3=prod_feat(phi_temp,phi_lin5,idx2);

idx3=randsample(2*m,m);
idx4=randsample(2*m,m);
phi_temp=sum_feat(phi1,phi2,idx3);
phi=sum_feat(phi_temp,phi3,idx4);
end