function [function_estimate,grad_estimate,runtime] = CosampGradEstimate(function_handle,x,cosamp_params,function_params)

% Uses CoSaMP to estimate a gradient using finite differences
% =================== INPUTS ===================================== %
% function_handle .............. function name (should be a .m file)
% x ............................ point at which to estimate gradient
% cosamp_params ................ Sampling matrix etc.
% function_params .............. (optional) parameters if required.
%
% ======================== OUTPUTS =============================== %
% function_estimate ........... value of function_handle(x)
% grad_estimate ............... estimated gradient

% Daniel Mckenzie, Hanqin Cai and Yuchen Lou 2019--2020
%


% == Unpack cosamp_params
maxiterations = cosamp_params.maxiterations;
Z = cosamp_params.Z;
delta = cosamp_params.delta;
sparsity = cosamp_params.sparsity;
tol = cosamp_params.tol;
num_samples = size(Z,1);


y = zeros(num_samples,1);
function_estimate = 0;

for i = 1:num_samples
    [y_temp1,~] = feval(function_handle,x + delta*Z(i,:)',function_params); % query at f(x+delta z_i)
    [y_temp2,~] = feval(function_handle,x,function_params); % query at f(x)
    function_estimate = function_estimate + y_temp2;
    y(i) = (y_temp1-y_temp2)/(sqrt(num_samples)*delta); % finite difference approximation to directional derivative.
end


Z = Z/sqrt(num_samples);

tic;
grad_estimate = cosamp(Z,y,sparsity,tol,maxiterations);
runtime = toc;
function_estimate = function_estimate/num_samples;
end
