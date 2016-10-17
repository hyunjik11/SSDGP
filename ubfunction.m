function [ub,naive_nld,nip_ub] = ubfunction(x,y,gp_var,precond)
    warning('off','all');
    xu = gp_var.X_u;
    m = size(xu,1);
    n = size(x,1);
    signal_var = gp_var.lik.sigma2;
    K_mn=gp_cov(gp_var,xu,x); K_mm=gp_trcov(gp_var,xu);
    L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
    L=L_mm'\K_mn; %L'*L=K_hat=K_mn'*(K_mm\K_mn)
    K_naive=L'*L;
    A=L*L'+signal_var*eye(m);
    L_naive=chol(A);
    naive_nld=(m-n)*log(signal_var)/2-sum(log(diag(L_naive)));
    [K,C]=gp_trcov(gp_var,x);
    switch precond
        case 'None'
            [~,~,~,~,~,cg_obj]=cgs_obj(C,y,[],m);
            nip_ub = min(cg_obj);
        case 'Nystrom'
            %function handle which gives (K_naive+signal_var*eye(n))\x
            myfun = @(w) (w-L'*(A\(L*w)))/signal_var;
            [~,~,~,~,~,pcg_obj]=cgs_obj(C,y,[],m,myfun);
            nip_ub = min(pcg_obj);
        case 'FIC'
            dinv=1./(diag(K)-diag(K_naive)+signal_var);
            Dinv=diag(dinv); %D=diag(K-K_naive)+signal_var*eye(n)
            Af=L*Dinv*L'+eye(m);
            %function handle which gives (K_fic+signal_var*eye(n))\x
            myfunf = @(w) (w-L'*(Af\(L*(w.*dinv)))).*dinv;
            [~,~,~,~,~,pcg_objf]=cgs_obj(C,y,[],m,myfunf);
            nip_ub = min(pcg_objf);
        case 'PIC'
            %funcion handle which gives (K_pic+signal_var*eye(n))\x
            [~,invfun]=blockdiag(K-K_naive,m,signal_var); %invfun is fhandle which gives (Kb+signal_var*eye(n))\w
            Ap=L*invfun(L')+eye(m);
            myfunp = @(w) invfun(w-L'*(Ap\(L*invfun(w))));
            [~,~,~,~,~,pcg_objp]=cgs_obj(C,y,[],m,myfunp);
            nip_ub = min(pcg_objp);
        otherwise
            error('precond is invalid')
    end
    ub = naive_nld + nip_ub - 0.5*n*log(2*pi);
end
