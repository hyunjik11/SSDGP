function [Kb,fun]=blockdiag(K,m,signal_var) %extract the ceil(n/m) diagonal blocks of K of size m by m or smaller
% also return function handle fun to compute M\x 
% for M=blockdiag(K)+signal_var*eye(n)
    n=size(K,1);
    num_blocks=ceil(n/m);
    Kb=zeros(n,n);
    for i=1:num_blocks
        Kb(m*(i-1)+1:min(m*i,n),m*(i-1)+1:min(m*i,n))=K(m*(i-1)+1:min(m*i,n),m*(i-1)+1:min(m*i,n));
    end
    fun = @(w) blockinv(Kb+signal_var*eye(n),num_blocks,m,w);
end
    