function [res,errest] = residueEst(fun,z0,varargin)
% residueEst: residue of fun at z0 with an error estimate, 1st or 2nd order pole
% usage: [res,errest] = residueEst(fun,z0)
% usage: [res,errest] = residueEst(fun,z0,prop1,val1,prop2,val2,...)
%
% ResidueEst computes the residue of a given function at a
% simple first order pole, or at a second order pole.
%
% The methods used by residueEst are polynomial extrapolants,
% which also yield an error estimate. The user can specify the
% method order, as well as the order of the pole. For more
% information on the exact methods, see the pdf file for my
% residueEst suite of codes.
%
% Finally, While I have not written this function for the
% absolute maximum speed, speed was a major consideration
% in the algorithmic design. Maximum accuracy was my main goal.
%
%
% Arguments (input)
%  fun - function to compute the residue for. May be an inline
%        function, anonymous, or an m-file. If there are additional
%        parameters to be passed into fun, then use of an anonymous
%        function is recommended.
%
%        fun should be vectorized to allow evaluation at multiple
%        locations at once. This will provide the best possible
%        speed. IF fun is not so vectorized, then you MUST set
%        'vectorized' property to 'no', so that residueEst will
%        then call your function sequentially instead.
%
%        Fun is assumed to return a result of the same
%        shape and size as its input.
%
%  z0  - scalar point at which to compute the residue. z0 may be
%        real or complex.
%
% Additional inputs must be in the form of property/value pairs.
%  Properties are character strings. They may be shortened
%  to the extent that they are unambiguous. Properties are
%  not case sensitive. Valid property names are:
%
%  'PoleOrder', 'MethodOrder', 'Vectorized' 'StepRatio', 'MaxStep'
%  'Path', 'DZ', 
%
%  All properties have default values, chosen as intelligently
%  as I could manage. Values that are character strings may
%  also be unambiguously shortened. The legal values for each
%  property are:
%
%  'PoleOrder' - specifies the order of the pole at z0.
%        Must be 1, 2 or 3.
%
%        DEFAULT: 1 (first order pole)
%
%  'Vectorized' - residueEst will normally assume that your
%        function can be safely evaluated at multiple locations
%        in a single call. This would minimize the overhead of
%        a loop and additional function call overhead. Some
%        functions are not easily vectorizable, but you may
%        (if your matlab release is new enough) be able to use
%        arrayfun to accomplish the vectorization.
%
%        When all else fails, set the 'vectorized' property
%        to 'no'. This will cause residueEst to loop over the
%        successive function calls.
%
%        DEFAULT: 'yes'
%
%  'Path' - Specifies the type of path to take the limit along.
%        Must be either 'spiral' or 'radial'. Spiral paths
%        will follow an exponential spiral into the pole, with
%        angular steps at pi/8 radians.
%
%        DEFAULT: 'radial'
%
%  'DZ' - Nominal step away from z0 taken in the estimation
%        All samples of fun will be taken at some path away
%        from zo, along the path z0 + dz. dz may be complex.
%        
%        DEFAULT: 1e8*eps(z0)
% 
%  'StepRatio' - ResidueEst uses a proportionally cascaded
%        series of function evaluations, moving away from your
%        point of evaluation along a path in the complex plane.
%        The StepRatio is the ratio used between sequential steps.
%
%        DEFAULT: 4
%
%
% See the document DERIVEST.pdf for more explanation of the
% algorithms behind the parameters of residueEst. In most cases,
% I have chosen good values for these parameters, so the user
% should never need to specify anything other than possibly
% the PoleOrder. I've also tried to make my code robust enough
% that it will not need much. But complete flexibility is in
% there for your use.
%
%
% Arguments: (output)
%  residue - residue estimate at z0.
%
%        When the residue is estimated as approximately zero,
%        the wrong order pole may have been specified.
%
%  errest - 95% uncertainty estimate around the residue, such that
%
%        abs(residue - fun(z0)*(z-z0)) < erest(j)
%
%        Large uncertainties here suggest that the wrong order
%        pole was specified for fun(z0).
%        
%
% Example:
%  A first order pole at z = 0
%
%  [r,e]=residueEst(@(z) 1./(1-exp(2*z)),0)
%
%  r =
%          -0.5
%
%  e =
%    4.5382e-12
%
% Example:
%  A second order pole around z = pi
%
%  [r,e]=residueEst(@(z) 1./(sin(z).^2),pi,'poleorder',2)
%
%  r =
%             1
%
%  e =
%    2.6336e-11
%
%
% See also: derivest, limest
%
% Author: John D'Errico
% e-mail: woodchips@rochester.rr.com
% Release: 1.0
% Release date: 3/27/2008

par.PoleOrder = 1;
% Always use 4 here
par.MethodOrder = [];
par.StepRatio = 4;
par.Vectorized = 'yes';
par.DZ = [];
par.Path = 'radial';
% 'DTheta' - Angle in radians for each subsequent step along
% a spiral path. This is only used when a spiral path is
% indicated.
par.DTheta = pi/8;

na = length(varargin);
if (rem(na,2)==1)
  error 'Property/value pairs must come as PAIRS of arguments.'
elseif na>0
  par = parse_pv_pairs(par,varargin);
end
par = check_params(par);

% Was fun a string, or an inline/anonymous function?
if (nargin<1)
  help residueEst
  return
elseif isempty(fun)
  error('fun was not supplied.')
elseif ischar(fun)
  % a character function name
  fun = str2func(fun);
end

% no default for z0
if (nargin<2) || isempty(z0)
  error('z0 was not supplied')
elseif numel(z0) > 1
  error('z0 must be scalar')
end

% supply a default step?
if isempty(par.DZ)
  if z0 == 0
    % special case for zero
    par.DZ = 1e8*eps(1);
  else
    par.DZ = 1e8*eps(z0);
  end
elseif numel(par.DZ)>1
  error('DZ must be scalar if supplied')
end

% MethodOrder will always = PoleOrder + 2
if isempty(par.MethodOrder)
  par.MethodOrder = par.PoleOrder+2;
end

% if a radial path
if (lower(par.Path(1)) == 'r')
  % a radial path. Just override any DTheta.
  par.DTheta = 0;
else
  % a spiral path
  % par.DTheta has a default of pi/8 (radians)
end

% Define the samples to use along a linear path
k = (-15:15)';
theta = par.DTheta*k;
delta = par.DZ*exp(sqrt(-1)*theta).*(par.StepRatio.^k);
ndel = length(delta);
Z = z0 + delta;

% sample the function at these sample points
if strcmpi(par.Vectorized,'yes')
  % fun is supposed to be vectorized.
  fz = fun(Z);
  fz = fz(:);
  if numel(fz) ~= ndel
    error('fun did not return a result of the proper size. Perhaps not properly vectorized?')
  end
else
  % evaluate in a loop
  fz = zeros(size(Z));
  for i = 1:ndel
    fz(i) = fun(Z(i));
  end
end

% multiply the sampled function by (Z - z0).^par.PoleOrder
fz = fz.*(delta.^par.PoleOrder);

% replicate the elements of fz into a sliding window
m = par.MethodOrder;
fz = fz(repmat((1:(ndel-m)),m+1,1) + repmat((0:m)',1,ndel-m));

% generate the general extrapolation rule
d = par.StepRatio.^((0:m)'-m/2);
A = repmat(d,1,m).^repmat(0:m-1,m+1,1);
[qA,rA] = qr(A,0);

% compute the various estimates of the prediction polynomials.
polycoef = rA\(qA'*fz);

% predictions for each model
pred = A*polycoef;
% and residual standard errors
ser = sqrt(sum((pred - fz).^2,1));

% the actual extrapolated estimates are just the first row of polycoef
% for a first order pole. For a second order pole, we need the first
% derivative, so we need the second row. Higher order poles are not
% estimable using this method due to numerical problems.
switch par.PoleOrder
  case 1
    residue_estimates = polycoef(par.PoleOrder,:);
  case 2
    % we need to scale the estimated parameters by delta, for each estimate
    residue_estimates = polycoef(par.PoleOrder,:)./delta(1:(end - par.MethodOrder)).';
    residue_estimates = residue_estimates*par.StepRatio.^(-par.MethodOrder/2);
    % also the error estimate
    ser = ser./delta(1:(end - par.MethodOrder)).' * par.StepRatio.^(-par.MethodOrder/2);
  case 3
    % we need to scale the estimated parameters by delta^(par.PoleOrder-1)
    residue_estimates = polycoef(par.PoleOrder,:)./delta(1:(end - par.MethodOrder)).'.^2;
    residue_estimates = residue_estimates*par.StepRatio.^(-2*par.MethodOrder/2);
    ser = ser./delta(1:(end - par.MethodOrder)).'.^2 * par.StepRatio.^(-2*par.MethodOrder/2);
end

% uncertainty estimate of the limit
rAinv = rA\eye(m);
cov1 = sum(rAinv.^2,2);

% 1 spare dof, so we use a student's t with 1 dof
errest = 12.7062047361747*sqrt(cov1(1))*ser;

% drop any estimates that were inf or nan.
k = isnan(residue_estimates) | isinf(residue_estimates);
errest(k) = [];
residue_estimates(k) = [];
% delta(k) = [];

% if nothing remains, then there was a problem.
% possibly the wrong order pole, or a bad dz.
nres = numel(residue_estimates);
if nres < 1
  error('Either the wrong order was specified for this pole, or dz was a very poor choice')
end

% sort the remaining estimates
[residue_estimates, tags] = sort(residue_estimates);
errest = errest(tags);
% delta = delta(tags);

% trim off the estimates at each end of the range
if nres > 4
  residue_estimates([1,end]) = [];
  errest([1,end]) = [];
  % delta([1,end]) = [];
end

% and take the one that remains with the lowest error estimate
[errest,k] = min(errest);
res = residue_estimates(k);
% delta = delta(k);

% for higher order poles, we need to divide by factorial(PoleOrder-1)
if par.PoleOrder>2
  res = res/factorial(par.PoleOrder-1);
  errest = errest/factorial(par.PoleOrder-1);
end

end % mainline end

% ============================================
% subfunction - check_params
% ============================================
function par = check_params(par)
% check the parameters for acceptability
%
% Defaults:
%
% par.PoleOrder = 1
% par.MethodOrder = []
% par.MaxStep = 1000;
% par.StepRatio = 2;
% par.Vectorized = 'yes'
% par.DZ = [];
% par.Path = 'radial'
% par.DTheta = pi/8;

% PoleOrder == 1 by default
if isempty(par.PoleOrder)
  par.PoleOrder = 1;
else
  if (length(par.PoleOrder)>1) || ~ismember(par.PoleOrder,[1 2 3 4 5])
    error 'PoleOrder must be 1, 2, or 3.'
  end
end

% MethodOrder == PoleOrder+2 by default
if isempty(par.MethodOrder)
  par.MethodOrder = par.PoleOrder + 2;
else
  if (length(par.MethodOrder)>1) || (par.MethodOrder<= par.PoleOrder)
    error 'MethodOrder must be at least PoleOrder+1.'
  end
end

% Path is char
valid = {'spiral', 'radial'};
if isempty(par.Path)
  par.Path = 'radial';
elseif ~ischar(par.Path)
  error 'Invalid Path: Must be character'
end
ind = find(strncmpi(par.Path,valid,length(par.Path)));
if (length(ind)==1)
  par.Path = valid{ind};
else
  error(['Invalid Path: ',par.Path])
end

% vectorized is char
valid = {'yes', 'no'};
if isempty(par.Vectorized)
  par.Vectorized = 'yes';
elseif ~ischar(par.Vectorized)
  error 'Invalid Vectorized: Must be character'
end
ind = find(strncmpi(par.Vectorized,valid,length(par.Vectorized)));
if (length(ind)==1)
  par.Vectorized = valid{ind};
else
  error(['Invalid Vectorized: ',par.Vectorized])
end

% DZ == [] by default
if (length(par.DZ)>1)
  error 'DZ must be empty or a scalar.'
end

end % check_params


% ============================================
% Included subfunction - parse_pv_pairs
% ============================================
function params=parse_pv_pairs(params,pv_pairs)
% parse_pv_pairs: parses sets of property value pairs, allows defaults
% usage: params=parse_pv_pairs(default_params,pv_pairs)
%
% arguments: (input)
%  default_params - structure, with one field for every potential
%             property/value pair. Each field will contain the default
%             value for that property. If no default is supplied for a
%             given property, then that field must be empty.
%
%  pv_array - cell array of property/value pairs.
%             Case is ignored when comparing properties to the list
%             of field names. Also, any unambiguous shortening of a
%             field/property name is allowed.
%
% arguments: (output)
%  params   - parameter struct that reflects any updated property/value
%             pairs in the pv_array.
%
% Example usage:
% First, set default values for the parameters. Assume we
% have four parameters that we wish to use optionally in
% the function examplefun.
%
%  - 'viscosity', which will have a default value of 1
%  - 'volume', which will default to 1
%  - 'pie' - which will have default value 3.141592653589793
%  - 'description' - a text field, left empty by default
%
% The first argument to examplefun is one which will always be
% supplied.
%
%   function examplefun(dummyarg1,varargin)
%   params.Viscosity = 1;
%   params.Volume = 1;
%   params.Pie = 3.141592653589793
%
%   params.Description = '';
%   params=parse_pv_pairs(params,varargin);
%   params
%
% Use examplefun, overriding the defaults for 'pie', 'viscosity'
% and 'description'. The 'volume' parameter is left at its default.
%
%   examplefun(rand(10),'vis',10,'pie',3,'Description','Hello world')
%
% params = 
%     Viscosity: 10
%        Volume: 1
%           Pie: 3
%   Description: 'Hello world'
%
% Note that capitalization was ignored, and the property 'viscosity'
% was truncated as supplied. Also note that the order the pairs were
% supplied was arbitrary.

npv = length(pv_pairs);
n = npv/2;

if n~=floor(n)
  error 'Property/value pairs must come in PAIRS.'
end
if n<=0
  % just return the defaults
  return
end

if ~isstruct(params)
  error 'No structure for defaults was supplied'
end

% there was at least one pv pair. process any supplied
propnames = fieldnames(params);
lpropnames = lower(propnames);
for i=1:n
  p_i = lower(pv_pairs{2*i-1});
  v_i = pv_pairs{2*i};
  
  ind = strmatch(p_i,lpropnames,'exact');
  if isempty(ind)
    ind = find(strncmp(p_i,lpropnames,length(p_i)));
    if isempty(ind)
      error(['No matching property found for: ',pv_pairs{2*i-1}])
    elseif length(ind)>1
      error(['Ambiguous property name: ',pv_pairs{2*i-1}])
    end
  end
  p_i = propnames{ind};
  
  % override the corresponding default in params
  params = setfield(params,p_i,v_i); %#ok
  
end

end % parse_pv_pairs






