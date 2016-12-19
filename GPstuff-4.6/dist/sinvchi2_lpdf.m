function y = sinvchi2_lpdf(x,nu,s2)
%SINVCHI2_LPDF Scaled log inverse-chi probability density function.
%
%   Description:
%   Y = SINVCHI2_LPDF(X,NU,S2) returns the scaled log inverse-chi2 probability
%   density function with parameters NU and S2, at the values in X.
%
%   The parameterization is as in Gelman, Carlin, Stern, Dunson, Vehtari,
%   and Rubin (2013). Bayesian Data Analysis, third edition.
%      NU is degrees of freedom
%      S2 is scale
%
% Copyright (c) 2003 Aki Vehtari

% This software is distributed under the GNU General Public 
% License (version 3 or later); please refer to the file 
% License.txt, included with the software, for details.

if nargin < 2, 
   error('Requires at least two input arguments.'); 
end

y = log(nu/2).*(nu/2) -gammaln(nu/2) + log(s2)/2*nu - log(x).*(nu/2+1) -nu.*s2/2./x;

