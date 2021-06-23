% ===================== Testing Block ZORO algorithm ================ %
% Test ZO-BCD-R and ZO-BCD-RC.
% Daniel McKenzie & Hanqin Cai 2019
% Yuchen Lou 2020-2021
% ================================================================== %

clear; close all; clc

rng(1);
% =================== Function and oracle parameters ================ %
function_params.D = 1000; % ambient dimension
s = 100; % sparsity
function_params.S = datasample(1:function_params.D,s,'Replace',false); % randomly choose support.
function_params.sigma = 0.01;  % noise level

% ================================ ZO-BCD Parameters ==================== %

ZOBCD_params.num_iterations = 50; % number of iterations
ZOBCD_params.delta1 = 0.0005;
ZOBCD_params.sparsity = s;
ZOBCD_params.step_size = 0.5;% Step size
ZOBCD_params.x0 = 100*randn(function_params.D,1) + 100;
%ZOBCD_params.init_grad_estimate = norm(4*ZOBCD_params.x0.^3);
ZOBCD_params.init_grad_estimate = 2*norm(ZOBCD_params.x0(function_params.S));
ZOBCD_params.max_time = 1e3;
ZOBCD_params.num_blocks = 5;
function_handle = "SparseQuadric";


ZOBCD_params.num_iterations = ZOBCD_params.num_blocks*ZOBCD_params.num_iterations;  % should be num_blocks* previous number.
% ====================== Run ZO-BCD-R ====================== %
ZOBCD_params.Type = "ZO-BCD-R";
ZOBCD_params.cosamp_max_iter = 10; % ceil(4*log(function_params.D));
[~,f_vals_r,~,~,num_samples_vec_r,~] = ZOBCD(function_handle,function_params,ZOBCD_params);

% === Plot
xx = cumsum(num_samples_vec_r);
semilogy(xx, abs(f_vals_r),'r*')
hold on

% ===================== Run ZO-BCD-RC ===================== %
ZOBCD_params.Type = "ZO-BCD-RC";
ZOBCD_params.cosamp_max_iter = 10;
[~,f_vals_rc,~,~,num_samples_vec_rc,~] = ZOBCD(function_handle,function_params,ZOBCD_params);

% == Plot
xx = cumsum(num_samples_vec_rc);
semilogy(xx,abs(f_vals_rc),'b*')

legend({'ZO-BCD-R', 'ZO-BCD-RC'})