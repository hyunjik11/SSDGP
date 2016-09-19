function [val,grad] = approx_ub_grad(w,gp,x,y)
% function to return approximate upper bound 
% 0.5*log(det(K_nyst+sigma_sq*eye(n))+0.5*y'*inv(K_pic + sigma_sq*eye(n))*y
% - logp(theta) +n/2*log(2*pi)
% and its grads wrt hyp
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
% Evaluate the contribution to val & grad of -NLD term + const
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
DCff = gp.lik.fh.cfg(gp.lik, x);
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
                sum(iLav'*sum(iAKuf'.*DKuf',2)); % -(3)
        end
        gprior = [gprior gprior_cf];
    end
end

% =================================================================
% Gradient of -NLD & -logp(theta) with respect to Gaussian likelihood function parameters
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
    gprior = [gprior gprior_lik];
end

% ============================================================
% Evaluate the contribution to val & grad of PIC -NIP term
% ============================================================
% get block indices
num_blocks=ceil(n/m);
ind=cell(1,num_blocks);
for i=1:num_blocks
    ind{i} = (m*(i-1)+1):min(m*i,n);
end
B=Luu\(K_fu'); 
iLaKfu = zeros(size(K_fu));  % f x u. Reset for calculating PIC -NIP
Labl=cell(1,length(ind)); % blocks of K-K_nyst
for i=1:length(ind)
    Qbl_ff = B(:,ind{i})'*B(:,ind{i});
    [~ , Cbl_ff] = gp_trcov(gp, x(ind{i},:));
    Labl{i} = Cbl_ff - Qbl_ff;
    iLaKfu(ind{i},:) = Labl{i}\K_fu(ind{i},:);
    edata = edata + 0.5*y(ind{i},:)'*(Labl{i}\y(ind{i},:));
end
A = K_uu+K_fu'*iLaKfu;
A = (A+A')./2;     % Ensure symmetry
[LA, notpositivedefinite] = chol(A,'upper');
if notpositivedefinite
    [grad, val] = set_output_for_notpositivedefinite;
    return
end
b = (y'*iLaKfu)/LA;
edata = edata - 0.5*(b*b'); % -NIP term for PIC

L = iLaKfu/LA;
b = zeros(1,n);
b_apu=(y'*L)*L';
for i=1:length(ind)
    b(ind{i}) = y(ind{i})'/Labl{i} - b_apu(ind{i});
end
iKuuKuf = Luu'\(Luu\K_fu');

if ~isempty(strfind(gp.infer_params, 'covariance'))
    % Loop over the  covariance functions
    i1=0;
    for i=1:ncf
        
        % Get the gradients of the covariance matrices
        % and gprior from gpcf_* structures
        gpcf = gp.cf{i};
        if savememory
            % If savememory option is used, just get the number of
            % hyperparameters and calculate gradients later
            np=gpcf.fh.cfg(gpcf,[],[],[],0);
        else
            DKuuc = gpcf.fh.cfg(gpcf, u); % cell of d/dth {K_uu}
            DKufc = gpcf.fh.cfg(gpcf, u, x);
            DKffc = cell(1,length(ind));
            for kk = 1:length(ind)
                DKffc{kk} = gpcf.fh.cfg(gpcf, x(ind{kk},:));
            end
            np=length(DKuuc); % number of hyperparams
        end
        
        for i2 = 1:np
            i1 = i1+1;
            if savememory
                DKuu=gpcf.fh.cfg(gpcf,u,[],[],i2);
                DKuf=gpcf.fh.cfg(gpcf,u,x,[],i2);
            else
                DKuu=DKuuc{i2};
                DKuf=DKufc{i2};
            end
            KfuiKuuKuu = iKuuKuf'*DKuu;
            %            H = (2*K_uf'- KfuiKuuKuu)*iKuuKuf;
            % Here we evaluate  gdata = -0.5.* (b*H*b' + trace(L*L'H)
            gdata(i1) = gdata(i1) - 0.5.*((2*b*DKuf'-(b*KfuiKuuKuu))*(iKuuKuf*b'));
            for kk=1:length(ind)
                if savememory
                    DKff=gpcf.fh.cfg(gpcf,x(ind{kk},:),[],[],i2);
                else
                    DKff=DKffc{kk}{i2};
                end
                gdata(i1) = gdata(i1) ...
                    + 0.5.*(-b(ind{kk})*DKff*b(ind{kk})' ...
                    + 2.*b(ind{kk})*DKuf(:,ind{kk})'*iKuuKuf(:,ind{kk})*b(ind{kk})'- ...
                    b(ind{kk})*KfuiKuuKuu(ind{kk},:)*iKuuKuf(:,ind{kk})*b(ind{kk})');
            end
        end
    end
end

% =================================================================
% Gradient of -NIP wrt Gaussian likelihood function parameters
% =================================================================
if ~isempty(strfind(gp.infer_params, 'likelihood')) && isfield(gp.lik.fh,'trcov')
    % Evaluate the gradient from Gaussian likelihood
    DCff = gp.lik.fh.cfg(gp.lik, x);
    for i2 = 1:length(DCff)
        i1 = i1+1;
        gdata(i1)= gdata(i1) -0.5*DCff{i2}.*b*b';
    end
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
