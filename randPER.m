function [z,M]=randPER(per,l,m)
%function to draw m random frequencies from spectral density of PER kernel
%p(x)=exp(-l^(-2))*sum_{n=-infty}^infty bessely(n,l^(-2))*delta(x-2*pi*n/per)
%first truncate density st tails have mass less than 2*epsilon=2*1e-4;
epsilon=1e-4;
%find M st exp(-l^(-2))*sum_{n=M+1}^infty bessely(n,l^(-2)) <epsilon
if 1/l^2>0.5
    temp=coth(1/l^2)-l^2;
else temp=tanh(1/l^2);
end
M=ceil((log(epsilon)+1/l^2+log(1-temp)-log(besseli(0,1/l^2)))/log(temp)-1);

%now we have guaranteed that tail mass of spectral density < 2*epsilon
%for|x|>2*pi*M/per
%use this to sample from discrete distrib
x=(-M:1:M)*2*pi/per;
p=besseli(-M:1:M,1/l^2)/exp(1/l^2);
if sum(p)<1-2*epsilon
    error('something wrong with code')
else p=p/sum(p); %normalise. sum(p) should be close to 1
end
z=discretesample(x',p',m);

end


