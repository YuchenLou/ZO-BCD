function [val,grad] = SparseQuadric(x_in,function_params)
%        val = SparseQuadric(x_in)
% Provides noisy evaluations of a sparse quadric of the form x^TQx + b^Tx
% here b is all ones.
%
% =========================== INPUTS ================================= %
% x_in ...................... Point at which to evaluate
% S ......................... Suppose set of sparse quadric. Keep this the
% same
% D ......................... Ambient dimension
% sigma ..................... sigma/sqrt(D) is per component Gaussian noise level
%
% ========================== OUTPUTS ================================== %
% 
% val ...................... noisy function evaluation at x_in
% grad ..................... exact (ie no noise) gradient evaluation at
% x_in
%
% Daniel Mckenzie
% 26th June 2019
%
 
% =========== Unpack function_params 
sigma = function_params.sigma;
S = function_params.S;
D = function_params.D;

noise = sigma*randn(1)./sqrt(D);
val = x_in(S)'*x_in(S) + noise;
grad = zeros(D,1);
grad(S) = 2*x_in(S);

end

