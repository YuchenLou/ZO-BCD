function [x_hat,f_vals,time_vec,gradient_norm,num_samples_vec,runtime] = ZOBCD(function_handle,function_params,ZOBCD_params)

% Basic Implementation of ZO-BCD with flexible sensing matrix.
% ======================== INPUTS ================================= %
% function_handle .......... name of oracle function.
% function_params .......... any parameters required by function
% ZOBCD_params .............. Parameters required by ZO-BCD.
% cosamp_params ............ Parameters required by the call to cosamp
%
% ======================== OUTPUTS =============================== %
% x_hat .................... final iterate.
% f_vals ................... vec containing f(x_k) for all k.
% time_vec ................. vec containing cumulative running time at each
% iteration.
% gradient_norm ............ vec containing ||g_k|| for all k.
% num_samples_vec .......... number of samples made at iteration k
%
% Daniel McKenzie 2019-2020, Yuchen Lou 2020-2021
%
x = ZOBCD_params.x0;
Type = ZOBCD_params.Type;
sparsity = ZOBCD_params.sparsity;
delta1 = ZOBCD_params.delta1;
grad_estimate = ZOBCD_params.init_grad_estimate;
num_iterations = ZOBCD_params.num_iterations;
step_size = ZOBCD_params.step_size;
max_time = ZOBCD_params.max_time;
D = length(x);
% =========== Initialize some vectors
f_vals = zeros(num_iterations,1);
time_vec = zeros(num_iterations,1);
gradient_norm = zeros(num_iterations,1);

% hard coding the following for now, can make a param. later if we want.
num_samples = 4*sparsity;
cosamp_params.maxiterations = ZOBCD_params.cosamp_max_iter;
cosamp_params.tol = 0.5;
cosamp_params.sparsity = sparsity;
oversampling_param = 1.5;

% ========== Initialize the sensing matrix

if isempty(ZOBCD_params.num_blocks)
    error('Number of blocks not specified')
else
    J = ZOBCD_params.num_blocks;
end

samples_per_block = ceil(oversampling_param*num_samples/J); block_size = D/J;
sparsity = ceil(oversampling_param*sparsity/J); % upper bound on sparsity per block.
cosamp_params.sparsity = sparsity;

if (Type == "ZO-BCD-R")
    % Block Rademacher Coordinate Descent
    Z = 2*(rand(samples_per_block,block_size) > 0.5) - 1;
    
elseif (Type == "ZO-BCD-RC")
    % Block Circulant Coordinate Descent
    z1 = 2*(rand(1,block_size) > 0.5) -1;
    Z1 = gallery('circul',z1);
    SSet = datasample(1:block_size,samples_per_block,'Replace',false);
    cosamp_params.SSet = SSet;
    
    z1_fft = fft(z1(:));
    cosamp_params.z1_fft = z1_fft;
    z_trans = Z1(:,1)';
    z_trans_fft = fft(z_trans(:));
    cosamp_params.z_trans_fft = z_trans_fft;
    Z = Z1(SSet,:);
end

cosamp_params.Z = Z;

% ========== Now do ZO-BCD
for i = 1:num_iterations
    tic
    %i
    cosamp_params.delta = delta1 * norm(grad_estimate);
    coord_index = randi(J);% randomly select a block
    block = (coord_index-1)*block_size+1 : coord_index*block_size;
    cosamp_params.block = block;
    [f_est,grad_estimate,runtime] = BlockCosampGradEstimate(function_handle,x,cosamp_params,function_params,Type);
    x = x - step_size*grad_estimate;
    f_vals(i) = f_est;
    num_samples_vec(i) = samples_per_block;
    if i==1
        time_vec(i) = toc;
    else
        time_vec(i) = time_vec(i-1) + toc;
    end
    if time_vec(i) >= max_time
        x_hat = x;
        % if max_time is reached, trim arrays by removing zeros
        f_vals = f_vals(f_vals ~= 0);
        time_vec = time_vec(time_vec ~=0);
        num_samples_vec = num_samples_vec(num_samples_vec~=0);
        disp('Max time reached!')
        return
    end
    if sparsity == 0
        break
    end
end


x_hat = x;
end

