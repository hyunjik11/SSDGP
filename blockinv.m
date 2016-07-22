function sol=blockinv(M,num_blocks,m,x)
    n=size(M,1);
    if numel(x)==max(size(x,1),size(x,2))
        sol=zeros(n,1);
        for i=1:num_blocks
            i1=m*(i-1)+1; i2=min(m*i,n);
            M_block=M(i1:i2,i1:i2);
            sol(i1:i2,:)=M_block\x(i1:i2);
        end
    else D=size(x,2);
        sol=zeros(n,D);
        for i=1:num_blocks
            i1=m*(i-1)+1; i2=min(m*i,n);
            M_block=M(i1:i2,i1:i2);
            sol(i1:i2,:)=M_block\x(i1:i2,:);
        end
    end
end