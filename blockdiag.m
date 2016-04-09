function Kb=blockdiag(K,m) %extract the ceil(n/m) diagonal blocks of K of size m by m or smaller
    n=size(K,1);
    num_blocks=ceil(n/m);
    Kb=zeros(n,n);
    for i=1:num_blocks
        Kb(m*(i-1)+1:min(m*i,n),m*(i-1)+1:min(m*i,n))=K(m*(i-1)+1:min(m*i,n),m*(i-1)+1:min(m*i,n));
    end
end
    