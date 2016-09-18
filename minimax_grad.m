function [val, grad] = minimax_grad(w, gp, x, y, alpha)
% function to return NEGATIVE maximin (i.e. minimax) objective and grads wrt hyp for fixed alpha
% alpha should be a col vector
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
% Evaluate the contribution to val & grad of -NLD term
% ============================================================
u = gp.X_u;
[Kv_ff, Cv_ff] = gp_trvar(gp, x);  % n x 1  vector
K_fu = gp_cov(gp, x, u);         % n x m
K_uu = gp_trcov(gp, u);          % m x m, noiseles covariance K_uu
K_uu = (K_uu+K_uu')./2;          % ensure the symmetry of K_uu
[Luu, notpositivedefinite] = chol(K_uu, 'lower');
if notpositivedefinite
    [val, grad] = set_output_for_notpositivedefinite;
    return
end
% Evaluate the Lambda (La)
% Q_ff = K_fu*inv(K_uu)*K_fu';
% Here we need only the diag(Q_ff), which is evaluated below
% B=Luu\(K_fu');       % m x n
% Qv_ff=sum(B.^2)';
Lav = Cv_ff-Kv_ff;   % n x 1, Vector of diagonal elements
iLaKfu = zeros(size(K_fu));  % f x u, diag(iLav)*K_fu = inv(La)*K_fu = K_fu/sigma_sq
for i=1:n
    iLaKfu(i,:) = K_fu(i,:)./Lav(i); 
end
A = K_uu+K_fu'*iLaKfu; % K_uu+K_uf*inv(La)*K_fu = K_uu+K_uf*K_fu/sigma_sq
A = (A+A')./2;     % Ensure symmetry
[LA, notpositivedefinite] = chol(A);
if notpositivedefinite
    [val, grad] = set_output_for_notpositivedefinite;
    return
end

iAKuf = LA\(LA'\K_fu'); % inv(A)*K_uf
L = iLaKfu/LA; % K_fu*inv(LA)/sigma_sq
%b = y'./Lav' - (y'*L)*L';
iKuuKuf = Luu'\(Luu\K_fu'); % inv(K_uu)K_uf
La = Lav;
%LL = sum(L.*L,2);
iLav=1./Lav; % 1/sigma_sq *ones(n,1)
%LL1=iLav-LL;

edata = sum(log(Lav))/2 - sum(log(diag(Luu))) + sum(log(diag(LA))) +n*log(2*pi)/2; % = -NLD + const
gdata = []; gprior = [];
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
            DKffc = gpcf.fh.cfg(gpcf, x, [], 1); % cell of d/dth {diag(K)} wrt loghyps th
            DKuuc = gpcf.fh.cfg(gpcf, u); % cell of d/dth {K_uu}
            DKufc = gpcf.fh.cfg(gpcf, u, x); % cell of d/dth {K_uf}
            DKc = gpcf.fh.cfg(gpcf, x); % cell of d/dth {K}
            np=length(DKuuc); % number of hyp in cov
        end
        gprior_cf = -gpcf.fh.lpg(gpcf);
        
        for i2 = 1:np
            i1 = i1+1;
            if savememory
                % If savememory option is used, just get the number of
                % hyperparameters and calculate gradients later
                % DKff=gpcf.fh.cfg(gpcf,x,[],1,i2);
                DKuu=gpcf.fh.cfg(gpcf,u,[],[],i2);
                DKuf=gpcf.fh.cfg(gpcf,u,x,[],i2);
                DKc=gpcf.fh.cfg(gpcf,x,[],[],i2);
            else
                % DKff=DKffc{i2};
                DKuu=DKuuc{i2};
                DKuf=DKufc{i2};
                DK=DKc{i2};
            end
            
            KfuiKuuDKuu = iKuuKuf'*DKuu; %K_fu*inv(K_uu)*dK_uu/dth
            gdata(i1) = -0.5*sum(iLav'*sum(iAKuf'.*KfuiKuuDKuu,2)) + ... % -((1)+(2))
                sum(iLav'*sum(iAKuf'.*DKuf',2)) - ... % -(3)
                0.5*alpha'*DK*alpha; % - dCG/dth
        end
        gprior = [gprior gprior_cf];
    end
end
% =================================================================
% Gradient with respect to Gaussian likelihood function parameters
% =================================================================
if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.lik.fh,'trcov')
    % Evaluate the gradient from Gaussian likelihood
    DCff = gp.lik.fh.cfg(gp.lik, x);
    gprior_lik = -gp.lik.fh.lpg(gp.lik);
    for i2 = 1:length(DCff)
        i1 = i1+1; 
        gdata(i1)= 0.5*sum(DCff{i2}./La-sum(L.*L,2).*DCff{i2}); %Dcff={lik.sigma2}
        % this is the gradient of data term wrt log sigma_sq
    end
    gcg = -0.5*sum(Lav.*(alpha.^2));
    gprior = [gprior gprior_lik+gcg];
end
    
% ============================================================
% Evaluate the contribution to val of CG term
% ============================================================
[~,C_ff] = gp_trcov(gp,x);
ecg = sum(alpha.*y) - 0.5*alpha'*C_ff*alpha;

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
val = edata + ecg + eprior;
grad = gdata + gprior;

function [val, grad] = set_output_for_notpositivedefinite()
  %instead of stopping to chol error, return NaN
  val = NaN;
  grad = NaN;
end

end