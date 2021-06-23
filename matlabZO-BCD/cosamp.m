function Sest = cosamp(Phi,u,K,tol,maxiterations)

% Cosamp algorithm
%   Input
%       K : sparsity of Sest
%       Phi : measurement matrix
%       u: measured vector
%       tol : tolerance for approximation between successive solutions.
%       maxiterations: maximal number of iterations allowed.
%   Output
%       Sest: Solution found by the algorithm
%
% Algorithm as described in "CoSaMP: Iterative signal recovery from 
% incomplete and inaccurate samples" by Deanna Needell and Joel Tropp.
% 


% This implementation was written by David Mary, 
% but modified 20110707 by Bob L. Sturm to make it much clearer,
% and corrected multiple times again and again.
% To begin with, see: http://media.aau.dk/null_space_pursuits/2011/07/ ...
% algorithm-power-hour-compressive-sampling-matching-pursuit-cosamp.html
%
% This script/program is released under the Commons Creative Licence
% with Attribution Non-commercial Share Alike (by-nc-sa)
% http://creativecommons.org/licenses/by-nc-sa/3.0/
% Short Disclaimer: this script is for educational purpose only.
% Longer Disclaimer see  http://igorcarron.googlepages.com/disclaimer

% Modified by Daniel McKenzie and HanQin Cai in 2020-2021 for a better
% performance.

% Initialization
Sest = zeros(size(Phi,2),1);
v = u;
t = 1; 
err = Inf;
numericalprecision = 1e-12;
T = [];
while (t <= maxiterations) && (err > tol)
  y = Phi'*v;
  [vals,Omega] = maxk(abs(y),2*K);
  Omega = Omega(vals > numericalprecision); %partial sorting
  T = union(Omega,T);
  b = Phi(:,T)\u;
  [vals,Kgoodindices] = maxk(abs(b),K); 
  Kgoodindices = Kgoodindices(vals > numericalprecision);
  T = T(Kgoodindices);
  Sest = zeros(size(Phi,2),1);
  b = b(Kgoodindices);
  Sest(T) = b;
  v = u - Phi(:,T)*b;
  
  t = t+1;
  err = norm(v)/norm(u);
end
