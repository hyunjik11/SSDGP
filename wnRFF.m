function phi=wnRFF(n,m,sf2)
%function to build hash kernel for RFF for wn kernel (i.e. identity)
h=randsample(m,n,true);
xi=randsample([-1,1],n,true);
phi=zeros(m,n);
for i=1:n
    j=h(i);
    phi(j,i)=xi(i);
end
phi=sqrt(sf2)*phi;
end

