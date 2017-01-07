function [val,grad,edata] = ub_grad(w,gp,x,y)
% function to return negative upper bound 
% 0.5*log(det(K_nyst+sigma_sq*eye(n))+0.5*y'*inv(K_nyst + sigma_sq*eye(n) + tr(K-K_nyst)*eye(n))*y
% - logp(theta) +n/2*log(2*pi)
% and its grads wrt log hyp
% gp is of type VAR
if ~all(isfinite(w(:)));
  % instead of stopping to error, return NaN
  val = NaN;
  grad = NaN;
  return
end
gp = gp_unpak(gp,w);
ncf = length(gp.cf);
n = size(x,1);

if ~strcmp(gp.type, 'VAR')
    error('GP not of type VAR')
end
if isfield(gp,'savememory') && gp.savememory
  savememory=1;
else
  savememory=0;
end

% ============================================================
% Evaluate the contribution to val of -NLD term + const
% ============================================================
u = gp.X_u; m = size(u,1);
[Kv_ff, Cv_ff] = gp_trvar(gp, x);  % n x 1  vector
K_fu = gp_cov(gp, x, u);         % n x m
K_uu = gp_trcov(gp, u);          % m x m, noiseles covariance K_uu
K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
[Luu, notpositivedefinite] = chol(K_uu); % L_uu'*L_uu = K_uu
if notpositivedefinite
    [val, grad] = set_output_for_notpositivedefinite;
    return
end
B=(Luu')\(K_fu');       % m x n, B'*B = K_hat := K_fu*inv(K_uu)*K_fu' (Nystrom)
Qv_ff=sum(B.^2);    % 1 x n, diag(K_hat)
Lav = Cv_ff-Kv_ff;   % n x 1, Vector of diagonal elements
signal_var = Lav(1);

iLaKfu = zeros(size(K_fu));  % f x u, diag(iLav)*K_fu = inv(La)*K_fu = K_fu/sigma_sq
for i=1:n
    iLaKfu(i,:) = K_fu(i,:)./Lav(i); 
end

A = B*B'/signal_var + eye(m);
A = (A+A')./2;     % Ensure symmetry
[LA, notpositivedefinite] = chol(A); 
if notpositivedefinite
    [val, grad] = set_output_for_notpositivedefinite;
    return
end
LA = LA*Luu; % LA'*LA = K_uu+K_uf*K_fu/signal_var
edata = 0.5*sum(log(Lav)) - sum(log(diag(Luu))) + sum(log(diag(LA))) ...
    +0.5*n*log(2*pi); % = -NLD + const

iAKuf = LA\(LA'\K_fu'); % inv(K_uu+K_uf*K_fu/signal_var)*K_uf
L = iLaKfu/LA; % K_fu*inv(LA)/signal_var 
% So L*L'=K_fu*inv(K_uu+K_uf*K_fu/signal_var)*K_uf/signal_var^2
% b = y'./Lav' - (y'*L)*L';
iKuuKuf = Luu\(Luu'\K_fu'); % inv(K_uu)*K_uf
La = Lav;
LL = sum(L.*L,2);
iLav=1./Lav; % 1/signal_var *ones(n,1)


% ============================================================
% Evaluate the contribution to val of -NIP term
% ============================================================
Lavub = Lav + sum(Kv_ff) - sum(Qv_ff); % diag(signal_var*I + tr(K-K_hat)*I)
c = Lavub(1);
Aub = B*B'/c + eye(m);
Aub = (Aub+Aub')./2;     % Ensure symmetry
[LAub, notpositivedefinite] = chol(Aub); 
if notpositivedefinite
    [val, grad] = set_output_for_notpositivedefinite;
    return
end
LAub = LAub*Luu; % LAub'*LAub = K_uu+K_uf*K_fu/c

bub=((y'*K_fu)/LAub)/c; 
edata = edata + 0.5*y'./Lavub'*y - 0.5*sum(bub.^2); % add on -NIP term

bbub = c * (LAub\(bub')); %inv(K_uu+K_uf*K_fu/c)*K_uf*y
temp1 = c*sum(bub.^2)-sum(y.^2); %(1/c)*y'*K_fu*inv(K_uu+K_uf*K_fu/c)*K_uf*y-y'*y
temp2 = sum(bbub.*(K_uu*bbub)); %y'*K_fu*inv(K_uu+K_uf*K_fu/c)*K_uu*inv(K_uu+K_uf*K_fu/c)*K_uf*y
% ============================================================
% Evaluate grad of -NLD term -NIP term + const wrt log kernel hyp
% ============================================================
gdata = [];
gprior = []; % Here it's better not to pre-allocate as gprior calculated per gpcf
if ~isempty(strfind(gp.infer_params, 'covariance'))
    % Loop over the covariance functions
    i1=0;
    for i=1:ncf
        % Get the gradients of the covariance matrices
        % and gprior from gpcf_* structures
        gpcf = gp.cf{i};
        if savememory
            np=gpcf.fh.cfg(gpcf,[],[],[],0);
        else
            DKuuc = gpcf.fh.cfg(gpcf, u); % cell of d/dth {K_uu}
            DKufc = gpcf.fh.cfg(gpcf, u, x); % cell of d/dth {K_uf}
            DKffc = gpcf.fh.cfg(gpcf, x,[],1); % cell of d/dth {diag(K)}
            np=length(DKuuc); % number of hyp in cov
        end
        gprior_cf = -gpcf.fh.lpg(gpcf);
        
        for i2 = 1:np
            i1 = i1+1;
            if savememory
                % If savememory option is used, just get the number of
                % hyperparameters and calculate gradients later
                DKff=gpcf.fh.cfg(gpcf,x,[],1,i2);
                DKuu=gpcf.fh.cfg(gpcf,u,[],[],i2);
                DKuf=gpcf.fh.cfg(gpcf,u,x,[],i2);
            else
                DKff=DKffc{i2};
                DKuu=DKuuc{i2};
                DKuf=DKufc{i2};
            end
            
            KfuiKuuDKuu = iKuuKuf'*DKuu; %K_fu*inv(K_uu)*dK_uu/dth
            
            % derivative of -NLD
            gdata(i1) = -0.5*sum(iLav'*sum(iAKuf'.*KfuiKuuDKuu,2)) + ... % -((1)+(2))
                sum(iLav'*sum(iAKuf'.*DKuf',2)); % -(3)
            
            % derivative of -NIP
            Dc = sum(DKff) - 2.*sum(sum(DKuf'.*iKuuKuf',2)) + ...
                 sum(sum(KfuiKuuDKuu.*iKuuKuf',2)); %d/dth {tr(K-K_hat)}
            gdata(i1) = gdata(i1) + 0.5*(1/c^2)*Dc*temp1 ...
                - (1/c^2)*sum((DKuf*y).*bbub) + 0.5*(1/c^3)*Dc*temp2 ...
                + 0.5*(1/c^2)*sum(bbub.*(DKuu*bbub)) + (1/c^3)*sum(bbub.*(DKuf*(K_fu*bbub)));
        end
        gprior = [gprior gprior_cf];
    end
end

% =================================================================
% Gradient of -NLD -NIP -logp(theta) wrt (log) Gaussian likelihood function parameters
% =================================================================
if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.lik.fh,'trcov')
    % Evaluate the gradient from Gaussian likelihood
    DCff = gp.lik.fh.cfg(gp.lik, x);
    gprior_lik = -gp.lik.fh.lpg(gp.lik);
    for i2 = 1:length(DCff)
        i1 = i1+1; 
        gdata(i1)= 0.5*sum(DCff{i2}./La-LL.*DCff{i2}) ... % derivative of -NLD - logp(theta)
            + signal_var*(0.5*(1/c^2)*temp1+ 0.5*(1/c^3)*temp2); % derivative of -NIP
        % this is the gradient of data term wrt log signal_var
    end
    gprior = [gprior gprior_lik];
end

% ============================================================
% Evaluate the prior contribution to val from covariance functions
% ============================================================
eprior = 0;
if ~isempty(strfind(gp.infer_params, 'covariance'))
  for i=1:ncf
    gpcf = gp.cf{i};
    eprior = eprior - gpcf.fh.lp(gpcf);
  end
end

% ============================================================
% Evaluate the prior contribution to val from Gaussian likelihood
% ============================================================
if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.lik.fh,'trcov') && isfield(gp.lik, 'p')
  % a Gaussian likelihood
  lik = gp.lik;
  eprior = eprior - lik.fh.lp(lik);
end
val = edata + eprior;
grad = gdata + gprior;

function [val, grad] = set_output_for_notpositivedefinite()
  %instead of stopping to chol error, return NaN
  val = NaN;
  grad = NaN;
end

end
