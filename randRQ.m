function [z,a]=randRQ(alpha,l,m)
%function to draw m random frequencies from spectral density of RQ kernel
%p(x)=temp*(temp*abs(x))^(alpha-0.5)*besselk(alpha-0.5,temp*abs(x))/(2^(alpha-0.5)*sqrt(pi)*gamma(alpha))
%first truncate density st tails have mass less than 2*epsilon=2*1e-4;
epsilon=1e-4;
%find x st besselk(alpha-0.5,x)*exp(x)/sqrt(x)<1. Note LHS is decreasing
x=1;
while besselk(alpha-0.5,x)>sqrt(x)*exp(-x)
    x=2*x;
end
%find a st a*sqrt(2*alpha)*l>x and
%gammainc(a*sqrt(2*alpha)*l,alpha+1,'upper')<2^(alpha-0.5)*sqrt(pi)*epsilon/alpha
temp=sqrt(2*alpha)*l;
a=x/temp;
while gammainc(a*temp,alpha+1,'upper')>2^(alpha-0.5)*sqrt(pi)*epsilon/alpha
    a=2*a;
end
%now we have guaranteed that tail mass of spectral density < 2*epsilon for|x|>a
%use this truncated pdf to sample from it using numerical inversion

xrange=linspace(-a,a,1000);
pdf=RQspec_pdf(alpha,l,xrange);
pdf=pdf/sum(pdf);
cdf=cumsum(pdf);
[cdf, mask] = unique(cdf); %remove duplicate elements
xrange=xrange(mask);
z=interp1(cdf,xrange,rand(m,1));
end


