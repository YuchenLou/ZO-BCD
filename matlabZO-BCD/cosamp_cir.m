function Sest = cosamp_cir(Phi,u,K,tol,maxiterations,z_trans_fft,SSet)

% Cosamp algorithm
%   Input
%       K : sparsity of Sest
%       Phi : measurement matrix
%       u: measured vector
%       tol : tolerance for approximation between successive solutions.
%       maxiterations: maximal number of iterations allowed.
%       z_trans_fft: pre-calculated fft.
%       SSet: index set of the samples selected.
%   Output
%       Sest: Solution found by the algorithm
%
% Algorithm as described in "CoSaMP: Iterative signal recovery from 
% incomplete and inaccurate samples" by Deanna Needell and Joel Tropp.
% 


% This implementation was based on the "cosamp" file by David Mary, 
% and modified 20110707 by Bob L. Sturm.
% It was further modified by Daniel Mckenzie & HanQin Cai for a better
% performance, and was modified by Yuchen Lou for the acceleration of fft, 
% in order to cope with the implementation of ZO-BCD-RC in 2020--2021

% Initialization
Sest = zeros(size(Phi,2),1);
v = u;
t = 1; 
err = Inf;
numericalprecision = 1e-12;
T = [];
while (t <= maxiterations) && (err > tol) 
  v1 = zeros(size(Phi,2),1); v1(SSet) = v;
  y = fft(z_trans_fft.*ifft(v1));
  
  [vals,Omega] = maxk(abs(y),2*K);
  Omega = Omega(vals > numericalprecision); 
  
  T = union(Omega,T);
  b = Phi(:,T)\u;
  %[b,flag] = lsqr(Phi(:,T),u);
  
  [vals,Kgoodindices] = maxk(abs(b),K);
  Kgoodindices = Kgoodindices(vals > numericalprecision);
  
  T = T(Kgoodindices);
  Sest = zeros(size(Phi,2),1);
  b = b(Kgoodindices);
  Sest(T) = b;
  v = u - Phi(:,T)*b; % Notice: The scale of this problem is small, and is
  % not necessary to be acclerated by fft in particular experiments.
  t = t+1;
  err = norm(v)/norm(u);
end
