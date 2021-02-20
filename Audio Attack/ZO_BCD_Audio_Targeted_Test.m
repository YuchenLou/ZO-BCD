% ====================== Wavelet Block Attack ========================= %
% This script attacks audio samples in a wavelet domain.
% Attack a large number of samples in order to determine the
% Attack Success Rate.
% This script only considers "left" as the source class, but you may change
% it to any other source class by modifying directory.
% Yuchen Lou & Daniel McKenzie 2020.8 - 2021.1
% ===================================================================== %

clear, close all, clc;

% ============== Load the network and images ================= %
load('commandNet.mat')
function_params.net = trainedNet;

sz = function_params.net.Layers(1).InputSize;
function_params.kappa = 0.1;
Classes = function_params.net.Layers(end).Classes; % list of all imagenet classes.

directory = 'Sounds/left'; % Select targeted attack audio path.
sounds = dir(fullfile(directory, '**/*.wav'));
num_sounds = length(sounds);
rng(1,'twister') % Fix the random seed
Order_of_Attack = randperm(num_sounds);
num_attack = 50; % number of attacks for each target label
num_attacked_sounds = 1; % Counter to keep track of how many images attacked.

% ================================ ZORO Parameters ==================== %
ZOBCD_params.num_iterations = 500; % number of iterations
ZOBCD_params.delta1 = 0.001;
ZOBCD_params.init_grad_estimate = 100;
ZOBCD_params.max_time = 3600;
ZOBCD_params.num_blocks = 6000;
ZOBCD_params.Type = "BCD";
function_handle = "AudioEvaluate";

% ============ Initialize vectors to keep track of success ============= %
%True_Labels = cell(num_attack,1);
Final_Labels = zeros(num_attack,10);
Attack_Success = zeros(num_attack,10);
Attack_Volume = zeros(num_attack,10);
ell_2_difference = zeros(num_attack,10);
ell_2_difference_wavelet = zeros(num_attack,10);
ell_0_difference_wavelet = zeros(num_attack,10);
Samples_to_success = zeros(num_attack,10);
Attacked_Sounds_Cell = cell(num_attack,10,3);
attacked_sounds_id = zeros(num_attack,10);

i = 1; % counter keeping track of which image we are currently considering.
while num_attacked_sounds <= num_attack
    flag = 0;
    while flag == 0
        ii = Order_of_Attack(i);
        [target_audio,fs] = audioread(fullfile(sounds(ii).folder, sounds(ii).name));
        
        % == Convert to spectrogram
        AuditorySpect = helperExtractAuditoryFeatures(target_audio,fs);
        
        [pred_label,scores] = classify(function_params.net,AuditorySpect);
        [~,temp_idx] = sort(scores,'descend');
        pred_idx = temp_idx(1);
        splitStr = regexp(sounds(ii).folder,'/','split');
        true_label = splitStr{end};
        true_idx = find(Classes == true_label);
        if pred_idx == true_idx
            flag = 1;
            disp(['Predicted label is ',pred_label])
            disp(['True label is ', true_label])
            disp('Commencing with attack')
        end
        i = i + 1;
    end
    function_params.true_id = true_idx;
    [target_audio_wavelet,~] = cwt(target_audio,'morse');
    function_params.target_audio_wavelet = target_audio_wavelet;
    function_params.label = true_label;
    %True_Labels{num_attacked_sounds} = true_label;
    disp(['Now attacking audio clip number ',num2str(ii)])
    
    % ====== Additional Parameters
    function_params.fs = fs;
    function_params.epsilon = 5; % Box Constraint params
    function_params.D = length(target_audio_wavelet(:));
    function_params.shape = size(target_audio_wavelet);
    ZOBCD_params.D = function_params.D;
    ZOBCD_params.sparsity = 0.025*ZOBCD_params.D;
    ZOBCD_params.step_size = 0.05; % Step size. 3e-4 is value used by Kaidi Xu
    ZOBCD_params.x0 = zeros(function_params.D,1);
    
    for iii = 1:10 % Targeted Attack
        if (iii ~= true_idx)
            disp(['Now attacking target id ',num2str(iii)])
            % ==== Set to targeted attack
            function_params.target_id = iii;
            if isnan(function_params.target_id) == 0
                function_params.target_label = function_params.net.Layers(end).ClassNames(function_params.target_id);
            end
            % ====================== run ZO-BCD Attack ======================= %
            outputs = ZOBCD_Adversarial_Attacks(function_handle,function_params,ZOBCD_params);
            
            % == Store attacked sound and noise
            Attacked_Sounds_Cell{num_attacked_sounds,iii,1} = target_audio; % store true sound
            Attacked_Sounds_Cell{num_attacked_sounds,iii,2} = outputs.Attacking_Noise;
            Attacked_Sounds_Cell{num_attacked_sounds,iii,3} = outputs.Attacked_Audio;
            % == Store distortion in wavelet domain
            ell_2_difference(num_attacked_sounds,iii) = norm(outputs.Attacking_Noise);
            ell_0_difference_wavelet(num_attacked_sounds,iii) = outputs.Wavelet_distortion_ell_0;
            ell_2_difference_wavelet(num_attacked_sounds,iii) = outputs.Wavelet_distortion_ell_2;
            % == Compute and store attack loudness
            % Change Attacking Noise!
            Attack_Volume(num_attacked_sounds,iii) = 20*log10(max(abs(outputs.Attacking_Noise))) - 20*log10(max(abs(target_audio)));
            % == Store Attack success parameters
            Attack_Success(num_attacked_sounds,iii) = outputs.Success;
            Final_Labels(num_attacked_sounds,iii) = outputs.Final_Label;
            Samples_to_success(num_attacked_sounds,iii) = sum(outputs.num_samples_vec);
        end
    end
    % don't forget to increment!
    num_attacked_sounds
    num_attacked_sounds = num_attacked_sounds + 1;
end

function_params.net = 'blank';  % clear this variable before saving
save('results_left.mat')