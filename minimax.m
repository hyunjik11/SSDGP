function [gp_var,old_val] = minimax(gp_var,x,y,opt)
    % optimizer to solve max_theta min_alpha{NLD + CG objective}
    warning('off','all');
    % gp_var (log)hyperparams
    w = gp_pak(gp_var);   
    
    % containers for new params and gp_var
    w_new = w; gp_var_new = gp_var;
    
    n = size(x,1);
    alpha = zeros(n,1);
    iter = 0;
    % initialise alpha
    alpha = pcg_step(gp_var,x,y,alpha);
    if sum(isnan(alpha))> 0
        return
    end
    new_val= minimax_grad(w,gp_var,x,y,alpha);
    old_val= new_val + 1e-6;
    while old_val > new_val
        iter = iter + 1;
        w = w_new;
        gp_var = gp_var_new;
        old_val = new_val;
        % Optimise wrt hyp using fminscg using function maximin_grad
        % that gives maximin objective and grads wrt hyp for fixed alpha.
        inner_loop=@(ww) minimax_grad(ww,gp_var,x,y,alpha);
        [w_new, ~]= fminscg(inner_loop,w,opt);
        gp_var_new = gp_unpak(gp_var,w_new);
        alpha = pcg_step(gp_var_new,x,y,alpha);
        if sum(isnan(alpha))> 0
            return
        end
        new_val = minimax_grad(w_new,gp_var_new,x,y,alpha);
        fprintf('iter=%d, old_val= %4.3f, new_val = %4.3f \n',iter, old_val, new_val)
    end

    function alpha = pcg_step(gp_var,x,y,alpha)
        xu=gp_var.X_u;
        m=size(xu,1);
        signal_var=gp_var.lik.sigma2;
        K_mn=gp_cov(gp_var,xu,x); K_mm=gp_trcov(gp_var,xu);
        L_mm=chol(K_mm); %L_mm'*L_mm=K_mm;
        L=L_mm'\K_mn; %L'*L=K_hat=K_mn'*(K_mm\K_mn)
        A=L*L'+signal_var*eye(m);
        L_naive=chol(A);
        [~,C]=gp_trcov(gp_var,x);
        %function handle which gives (K_naive+signal_var*eye(n))\x
        myfun = @(ww) (ww-L'*(A\(L*ww)))/signal_var;
        [alpha,flag,~,~,~,~]=cgs_obj(C,y,[],[],myfun,[],alpha); %gives alpha as col vec
        switch flag
            case 0
                fprintf('PCG has converged in m iter\n')
            case 1
                fprintf('PCG has not converged in m iter\n')       
            case 2
                fprintf('Preconditioner in PCG ill-conditioned \n')
                alpha = nan;
                return
            case 3
                fprintf('PCG stagnated (Two consecutive iterates the same \n')
                alpha = nan;
                return
            case 4
                fprintf('One of the scalar quantities calculated during cgs became too small/large. \n')
                alpha = nan;
                return
        end
    end

end
    