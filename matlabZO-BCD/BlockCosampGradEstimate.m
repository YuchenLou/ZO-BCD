function [function_estimate,grad_estimate,runtime] = BlockCosampGradEstimate(function_handle,x,cosamp_params,function_params,Type)

% Uses CoSaMP to estimate a (partial) gradient using finite differences
% Block coordinate descent edition
% =================== INPUTS ===================================== %
% function_handle .............. function name (should be a .m file)
% x ............................ point at which to estimate gradient
% cosamp_params ................ Sampling matrix etc.
% function_params .............. (optional) parameters if required.
% Type ......................... Choice of ZO-BCD-R or -RC
%
% ======================== OUTPUTS =============================== %
% function_estimate ........... value of function_handle(x)
% grad_estimate ............... estimated gradient
% runtime ..................... recording runtime of a single cosamp

% Daniel Mckenzie, Hanqin Cai and Yuchen Lou 2019--2021
%


% == Unpack cosamp_params
maxiterations = cosamp_params.maxiterations;
Z = cosamp_params.Z;
delta = cosamp_params.delta;
sparsity = cosamp_params.sparsity;
tol = cosamp_params.tol;
block = cosamp_params.block;
num_samples = size(Z,1);  % number of samples
dim = length(x);
%block_size = length(block);

% === Pad Z with zeros to feed into oracle
Z_padded = zeros(num_samples,dim);
Z_padded(:,block) = Z;


y = zeros(num_samples,1);
function_estimate = 0;

for i = 1:num_samples
    [y_temp1,~] = feval(function_handle,x + delta*Z_padded(i,:)',function_params); % query at f(x+delta z_i)
    [y_temp2,~] = feval(function_handle,x,function_params); % query at f(x)
    function_estimate = function_estimate + y_temp2;
    y(i) = (y_temp1-y_temp2)/(sqrt(num_samples)*delta); % finite difference approximation to directional derivative.
end


Z = Z/sqrt(num_samples);

if (Type == "ZO-BCD-RC")
    %z1_fft = cosamp_params.z1_fft;
    z_trans_fft = cosamp_params.z_trans_fft;
    SSet = cosamp_params.SSet;
    
    tic;
    block_grad_estimate = cosamp_cir(Z,y,sparsity,tol,maxiterations,z_trans_fft,SSet);
    runtime = toc;
    
else
    tic;
    block_grad_estimate = cosamp(Z,y,sparsity,tol,maxiterations);
    runtime = toc;
end

grad_estimate = zeros(dim,1);
grad_estimate(block) = block_grad_estimate;
function_estimate = function_estimate/num_samples;
end
