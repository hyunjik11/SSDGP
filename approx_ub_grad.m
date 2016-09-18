function [val,grad] = approx_ub_grad(w,gp,x,y)
% function to return approximate upper bound and grads wrt hyp
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
u = gp.X_u; m = size(u,1);
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
% ============================================================
% Evaluate the contribution to val & grad of PIC -NIP term
% ============================================================
% get block indices
ind={};
num_blocks=ceil(n/m);
for i=1:num_blocks
    ind{i} = (m*(i-1)+1):min(m*i,n);
end
B=Luu\(K_fu'); 
iLaKfu = zeros(size(K_fu));  % f x u. Reset for calculating PIC -NIP
Labl={}; % blocks of K-K_nyst
for i=1:length(ind)
    Qbl_ff = B(:,ind{i})'*B(:,ind{i});
    [~ , Cbl_ff] = gp_trcov(gp, x(ind{i},:));
    Labl{i} = Cbl_ff - Qbl_ff;
    iLaKfu(ind{i},:) = Labl{i}\K_fu(ind{i},:);
    [~, notpositivedefinite]=chol(Labl{i},'upper');
    if notpositivedefinite
        [grad, val] = set_output_for_notpositivedefinite;
        return
    end
    edata = edata + 0.5*y(ind{i},:)'*(Labl{i}\y(ind{i},:));
end
A = K_uu+K_fu'*iLaKfu;
A = (A+A')./2;     % Ensure symmetry
[A, notpositivedefinite] = chol(A,'lower');
if notpositivedefinite
    [grad, val] = set_output_for_notpositivedefinite;
    return
end
b = (y'*iLaKfu)*inv(A)';
edata = edata - 0.5* b*b'; % -NIP term for PIC
