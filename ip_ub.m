function [ub,relres,resvec]=ip_ub(K,y,signal_var,maxiter,PC)
%function to give an upper bound to the -innerprod/2 term for log marginal
%PC is an optional preconditioner
%use default tolerance for cgs which is 1e-6
%we also give relres:relative residual norm norm(b-A*x)/norm(b)
%and resvec:residual norms at each cg iter including norm(b-A*x0)
    n=length(y);
    C=K+signal_var*eye(n);
    if nargin==4 %CG
        [sol,~,relres,~,resvec]=cgs(C,y,[],maxiter);
    else %PCG
        [sol,~,relres,~,resvec]=pcg(C,y,[],maxiter,PC);
    end
    ub=sol'*(C*sol)/2-sol'*y;
end

        