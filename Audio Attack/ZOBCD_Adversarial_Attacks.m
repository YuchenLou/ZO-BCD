function outputs = ZOBCD_Adversarial_Attacks(function_handle,function_params,ZORO_params)

% Basic Implementation of ZO-BCD designed to be used specifically 
% for audio attacks.
% 
% ======================== INPUTS ================================= %
% function_handle .......... name of oracle function.
% function_params .......... any parameters required by function
% ZORO_params .............. Parameters required by ZORO.
% cosamp_params ............ Parameters required by the call to cosamp
%
% ======================== OUTPUTS =============================== %
% x_hat .................... final iterate.
% f_vals ................... vec containing f(x_k) for all k.
% time_vec ................. vec containing cumulative running time at each
% iteration.
% gradient_norm ............ vec containing ||g_k|| for all k.
% num_samples_vec .......... number of samples made at iteration k
% I_attack ................. image after the attack
% iter ..................... number of iteration for a successful attack
%
% Yuchen Lou, Daniel McKenzie 2020 - 2021
%

D = ZORO_params.D;
sparsity = ZORO_params.sparsity;
num_iterations = ZORO_params.num_iterations;
Type = ZORO_params.Type;
delta1 = ZORO_params.delta1;
%grad_estimate = ZORO_params.init_grad_estimate;
x = ZORO_params.x0;
step_size = ZORO_params.step_size;
max_time = ZORO_params.max_time;
Wavelet_distortion_ell_0 = NaN;
Wavelet_distortion_ell_2 = NaN;

iter = num_iterations;
Final_Label = function_params.true_id;

% =========== Initialize some vectors
f_vals = zeros(num_iterations,1);
time_vec = zeros(num_iterations,1);
Success = 0; % flag to check whether attack was succesfull or not.

% hard coding the following for now, can make a param. later if we want.
%num_samples = ceil(4*sparsity*log(D));
cosamp_params.maxiterations = 50;
cosamp_params.tol = 1e-2;
cosamp_params.sparsity = sparsity;
oversampling_param = 1.1;

% =========== Initialize sensing matrix
J = ZORO_params.num_blocks;
block_size = ceil(D/J) - 1;
sparsity = ceil(oversampling_param*sparsity/J); % upper bound on sparsity per block.
samples_per_block = ceil(sparsity*log(block_size))
cosamp_params.sparsity = sparsity;

if (Type == "BCD")
    % Block Rademacher Coordinate Descent
    Z = 2*(rand(samples_per_block,block_size) > 0.5) - 1;

elseif (Type == "BCCD")
    % Block Circulant Coordinate Descent
    z1 = 2*(rand(1,block_size) > 0.5) -1;
    Z1 = gallery('circul',z1);
    SSet = datasample(1:block_size,samples_per_block,'Replace',false);
    Z = Z1(SSet,:);
end

cosamp_params.Z = Z;
cosamp_params.delta = delta1;
% ========== Now do ZORO
% This code only allows for block methods
    
% ==== Initialize the blocks. 
% Using random blocks here, but we can experiment.
P = randperm(function_params.D);


for i = 1:num_iterations
    tic
    disp(['Number of iterations = ',num2str(i)])
    coord_index = randi(J-1);% randomly select a block HACK: avioding the last block which may be smaller
    block = P((coord_index-1)*block_size+1:coord_index*block_size);
    cosamp_params.block = block;
    [f_est,grad_estimate] = BlockCosampGradEstimate(function_handle,x,cosamp_params,function_params);
    x = x - step_size*grad_estimate;
    % Box Constraint
    %x(x > function_params.epsilon) = function_params.epsilon;
    %x(x < -function_params.epsilon) = -function_params.epsilon;
    f_vals(i) = f_est;
    Attacking_Noise = reshape(x,function_params.shape);
    Attacked_audio = function_params.target_audio_wavelet + Attacking_Noise;
    num_samples_vec(i) = samples_per_block;
    [val,new_label,scores] = AudioEvaluate(x,function_params);
    disp(new_label);
    disp(['Current loss is ',num2str(val)])
    disp(scores)
    if isnan(function_params.target_id)
        if new_label ~= function_params.label
            iter = i;
            disp('Attack succesful')
            Final_Label = new_label;
            Success = 1;
            Wavelet_distortion_ell_0 = nnz(x);
            Wavelet_distortion_ell_2 = norm(x,2);
            break
        end
    else
        if new_label == function_params.target_label
            iter = i;
            disp('Attack succesful')
            Final_Label = new_label;
            Success = 1;
            Wavelet_distortion_ell_0 = nnz(x);
            Wavelet_distortion_ell_2 = norm(x,2);
            break
        end
    end
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
        % Now pack all the outputs into a struct.
        
        outputs.Attacking_Noise = icwt(Attacking_Noise,'morse');
        outputs.Attacked_Audio = icwt(Attacked_audio,'morse');
        outputs.f_vals = f_vals;
        outputs.iter = iter;
        outputs.num_samples_vec = num_samples_vec;
        outputs.Success = Success;
        outputs.Final_Label = Final_Label;
        Wavelet_distortion_ell_0 = nnz(x);
        Wavelet_distortion_ell_2 = norm(x,2);
        outputs.Wavelet_distortion_ell_0 = Wavelet_distortion_ell_0;
        outputs.Wavelet_distortion_ell_2 = Wavelet_distortion_ell_2;

        return
    end
    if sparsity == 0
        break
    end
end

% Now pack all the outputs into a struct.
outputs.Attacking_Noise = icwt(Attacking_Noise,'morse');
outputs.Attacked_Audio = icwt(Attacked_audio,'morse');
outputs.f_vals = f_vals;
outputs.iter = iter;
outputs.num_samples_vec = num_samples_vec;
outputs.Success = Success;
outputs.Final_Label = Final_Label;
outputs.Wavelet_distortion_ell_0 = Wavelet_distortion_ell_0;
outputs.Wavelet_distortion_ell_2 = Wavelet_distortion_ell_2;

end

