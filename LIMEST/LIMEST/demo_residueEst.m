%% How does residueEst work?

%{
How might one best compute the residue of a complex function at a known
pole of low order using numerical techniques? Since the function is not
finite at the pole, we cannot evaluate it directly there.

The approach taken by residueEst is to sample the function at a geometric
sequence of points approaching the pole. This is not unlike the approach
I took with my derivest tools.

For example, suppose we wish to compute the residue of the first order
pole of 1./sin(z), at z = pi. (The residue is -1.) For a first order
pole, the residue is given by the limit:

  lim     (f(z)*(z-z0))
z --> z0

ResidueEst solves this problem by choosing some offset, dz, from z0.
Pick some small offset, say 1e-16. Now, evaluate the function at a
sequence of points z0 + dz, z0 + k*dz, z0 + k^2*dz, ..., z0 + k^n*dz.

The default in derivest is to use the geometric factor of k = 4. 

%}

format long g
z0 = pi;
dz = 1e-16;
k = 4;
delta = dz*k.^(0:30)

% Now, evaluate the function at each of these points, z0 + delta, and
% multiply by (z0 + delta - z0) == delta.
fun = @(z) 1./sin(z);
fun_of_z = fun(z0+delta).*delta

%% Use a polynomial to fit a subsequence of points. Polyfit would do.
% Note that if we look at the points that are very close to z0, then the
% polynomial may have strange coefficients.

% For a first order pole, we are really only interested in the value
% of the constant terms for this polynomial model. Essentially, this
% is the extrapolated prediction of the limiting value of our product
% extrapolated down to delta == 0.
P0 = polyfit(delta(1:4),fun_of_z(1:4),2)

% A nice feature of this sequence of points, is we can re-use the
% function values, building a sliding sequence of models.
P0 = polyfit(delta(2:5),fun_of_z(2:5),2)
P0 = polyfit(delta(3:6),fun_of_z(3:6),2)
P0 = polyfit(delta(4:7),fun_of_z(4:7),2)
P0 = polyfit(delta(5:8),fun_of_z(5:8),2)

% See how, as we move along this sequence using a sliding window of 4
% points, the constant terms is approaching -1.
P0 = polyfit(delta(5:8),fun_of_z(5:8),2)
P0 = polyfit(delta(12:15),fun_of_z(12:15),2)

% At some point, we move just too far away from the pole, and our
% extrapolated estimate at the pole becomes poor again.
P0 = polyfit(delta(26:29),fun_of_z(26:29),2)
P0 = polyfit(delta(27:30),fun_of_z(27:30),2)

%% The trick is learn from Goldilocks.
% Choose a prediction of the pole for some model that is not too
% close to the pole, nor to far away. The choice is made by a simple
% set of rules. First, discard any predictions of the limit that are
% either NaN or inf. Then trim off a few more predictions, leaving
% only those predictions in the middle. Next, each prediction is made
% from a polynomial model with ONE more data point than coefficients
% to estimate. This yields an estimate of the uncertainty in the
% constant term using standard statistical methodologies. While a
% 95% confidence limit has no true statistical meaning, since the
% data is not truly random, we can still use the information.
%
% ResidueEst choose the model that had the narrowest confidence
% bounds around the constant term.
[res,errest] = residueEst(@(z) 1./sin(z),pi)

% See that if the true residue is really is -1, then we should have
% (approximately)
%
% abs(res - (-1)) < errest
[abs(res - (-1)),errest]

%% Residue of exp(z)/z, around z0 == 0
% Here the residue should be exp(0) == 1
[res,errest] = residueEst(@(z) exp(z)./z,0)

%% A second order pole, exp(3*z)./z.^2
% The residue should be 3.
[res,errest] = residueEst(@(z) exp(3*z)./z.^2,0,'poleorder',2)

% The error estimate is a reaonable one here too.
[abs(res - 3),errest]

%% A second order pole, 1./z.^2, at z0 = 0
% The residue should be 0
[res,errest] = residueEst(@(z) 1./z.^2,0,'poleorder',2)

%% Another 2nd order pole with a zero residue, 1./sin(z).^2, z0 = pi
% The residue here is zero
[res,errest] = residueEst(@(z) 1./(sin(z).^2),pi,'poleorder',2)

