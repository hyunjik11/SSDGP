function px=RQspec_pdf(alpha,l,x)
%function to compute value of RQ spectral density at x
%for x=0, px is defined as the limit. Need symbolic toolbox for this
%x can be a vector
temp=sqrt(2*alpha)*l;
px=temp*(temp.*abs(x)).^(alpha-0.5).*besselk(alpha-0.5,temp*abs(x))/(2^(alpha-0.5)*sqrt(pi)*gamma(alpha));
idx=find(isinf(px)|isnan(px));
for i=idx
    p=@(v) temp*(temp.*abs(v)).^(alpha-0.5).*besselk(alpha-0.5,temp*abs(v))/(2^(alpha-0.5)*sqrt(pi)*gamma(alpha));
    px(i)=limest(p,x(i));
end
end