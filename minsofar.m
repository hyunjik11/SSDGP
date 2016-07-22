function h=minsofar(x)
    n=length(x);
    h=zeros(n,1);
    minthusfar=x(1);
    for i=1:n
        minthusfar=min(minthusfar,x(i));
        h(i)=minthusfar;
    end
end
