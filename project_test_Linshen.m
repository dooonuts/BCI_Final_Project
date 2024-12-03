clear;
project_data_folder = "C:\lls_university\Linshen_pc\UT_Austin\fall2024\ECE385JBIOENG\final_project\BCI Project Data\";
allFiles = dir(fullfile(project_data_folder, '**', '*.gdf'));
gdfFiles = fullfile({allFiles.folder}, {allFiles.name})';
% Ensure output is a 1D cell array (transpose if necessary)
gdfFiles = gdfFiles(:);

repetitions = ['r001';'r002';'r003';'r004'];
sessions = ['s001';'s002';'s003'];
subjects = ['107';'108';'109'];

all_sessions = create_classes(gdfFiles);

% Filtering Sessions
MI_sessions = {};
[~, num_sessions] = size(all_sessions);
for i=1:num_sessions
    if (convertCharsToStrings(all_sessions{i}.Type) == "MI")
        curr_session = all_sessions{i};
        [pe_rest, pe_rest_spectrum, pe_rest_famp, rest_tags, pe_mi, pe_mi_spectrum, pe_mi_famp, mi_tags] = preprocess_session(curr_session.Filename);
        curr_session.PE_MI = pe_mi;
        curr_session.PE_MI_Spectrum = pe_mi_spectrum;
        curr_session.PE_MI_Famp = pe_mi_famp;
        curr_session.MI_Tags = mi_tags;

        curr_session.PE_Rest = pe_rest;
        curr_session.Rest_Tags = rest_tags;
        curr_session.PE_Rest_Spectrum = pe_rest_spectrum;
        curr_session.PE_Rest_Famp = pe_rest_famp;
        MI_sessions{end+1} = curr_session;
    end
end
%% Testing 
% first_trial = curr_session.PE_MI{1};
% first_trial_spectrum = curr_session.PE_MI_Spectrum;
% first_trial_famp = curr_session.PE_MI_Famp;
% figure(1); clf;
% plot(first_trial);
% figure(2); clf;
% plot(first_trial_spectrum{1},first_trial_famp{1});
MI_sessions{end+1} = curr_session; 



% offline_mi_sessions = {};
% online_mi_sessions = {};
% smallest_array = 8000;
% [~, num_mi_sessions]=size(MI_sessions);
% for i=1:num_mi_sessions
%    if(convertCharsToStrings(MI_sessions))
% end 

%% TODO LIST
% Finish PCA
% Create Model and Test
% Linear Discriminant Analysis/Linear Regression
% What happens if we train with first only session of top of that?

% 1 PCA demo 
data = rand(240, 192000); % 
% Step 1: std
data_standardized = zscore(data);
% Step 2: 
[coeff, score, latent, tsquared, explained] = pca(data_standardized);

% coeff: The coefficient matrix of the principal components, each column is a principal component
% score: Projection of original data onto principal components
% latent: The eigenvalue of each principal component indicates the amount of variance explained
% explained: The proportion of variance explained by each principal component (percentage)

% Step 3: Analyze principal components
% View the cumulative proportion of variance explained by the first 10 principal components
cumsum_explained = cumsum(explained); % Cumulative proportion of variance explained
disp('The cumulative variance ratio of the first 10 principal components is:');
disp(cumsum_explained(1:10));

% Decide how many principal components to keep, for example, keep those that explain more than 95% of the variance
num_components = find(cumsum_explained >= 95, 1);
disp(['number of component : ', num2str(num_components)]);

% Step 4: PCA result
reduced_data = score(:, 1:num_components);

% Step 5: Data Visulization
% to PCA 
if num_components >= 2
    figure;
    scatter(reduced_data(:, 1), reduced_data(:, 2), 50, 'filled');
    xlabel('Principal Component 1');
    ylabel('Principal Component 2');
    title('PCA: First Two Principal Components');
    grid on;
end

% 3D visulization 
if num_components >= 3
    figure;
    scatter3(reduced_data(:, 1), reduced_data(:, 2), reduced_data(:, 3), 50, 'filled');
    xlabel('Principal Component 1');
    ylabel('Principal Component 2');
    zlabel('Principal Component 3');
    title('PCA: First Three Principal Components');
    grid on;
end

% Step 6: save the data
% save('reduced_data.mat', 'reduced_data');

%% 2 PCA feature distribution visulization 
reduced_data_200 = reduced_data(:,1:200); % here is the after PCA data 
% variable 1 and variable 2 belongs t0 two different class 0 and 1.
first_200_samples = reduced_data_200(1:120, :); % first 200 sample 
last_200_samples = reduced_data_200(121:240, :); % last 200 sample 

% combine the 200 sample as a matrix 
% combined_samples = [first_200_samples, last_200_samples];
combined_samples = [];
for i = 1:200 % feature number 
    combined_samples = [combined_samples; first_200_samples(:, i)'; last_200_samples(:, i)'];
end
combined_samples = combined_samples'

% boxplot
figure(5); 
boxplot(combined_samples, 'Labels', arrayfun(@(x) ['Variable ', num2str(x)], 1:400, 'UniformOutput', false));

% legend and title 
legend({'Class 0', 'Class 1 '}, 'Location', 'best'); % the legned does not shows clearly 
title('Comparison of First 200 vs Last 200 Samples');












%% EOG Artifact Removal - Regression

%Source: Schlogl et. al. Clinical Neuropsychology 2007
%Calculate EOG weights using the cross-covariance matrices
% 
% b = inv(EOG'*EOG)*(EOG'*dataTempFilt);
% 
% dataTempEOG = dataTempFilt - EOG*b;




%% PCA 
% coeff = pca(dataSpaceTempFilt);

%% Model 

% Preprocess Each Session and Return, 
function [pe_rest, pe_rest_spectrum, pe_rest_famp, rest_tags, pe_mi, pe_mi_spectrum, pe_mi_famp, mi_tags] = preprocess_session(curr_session_file)

    [s,h] = sload(curr_session_file);
    
    % Certain channels are unused:
    s = s(:,1:34);

    [restMatrix,rest_tags,miMatrix,mi_tags] = crop_sort_signals(s,h);

    [~, num_rest_trials] = size(restMatrix);
    [~, num_mi_trials] = size(miMatrix);
    pe_rest = {};
    pe_rest_spectrum = {};
    pe_rest_famp = {};
   
    pe_mi = {};
    pe_mi_spectrum = {};
    pe_mi_famp = {};
    for i=1:num_rest_trials
        [pe_rest{i},pe_rest_spectrum{i},pe_rest_famp{i}] = preprocess_trial(restMatrix{i});
    end
    for i=1:num_mi_trials
        [pe_mi{i},pe_mi_spectrum{i},pe_mi_famp{i}] = preprocess_trial(miMatrix{i});
    end
    
end
% preprocess_session_with_padding, Linshen (Daniel said he is doing so in last version)
function [pe_rest, pe_rest_spectrum, pe_rest_famp, rest_tags, pe_mi, pe_mi_spectrum, pe_mi_famp, mi_tags] = preprocess_session_with_padding(curr_session_file)
    [s,h] = sload(curr_session_file);
    
    % from 1 to 34:
    s = s(:,1:34);

    [restMatrix,rest_tags,miMatrix,mi_tags] = crop_sort_signals(s,h);

    % find all maximum latency 
    max_rest_length = max(cellfun(@(x) size(x, 1), restMatrix));
    max_mi_length = max(cellfun(@(x) size(x, 1), miMatrix));

    % 对 restMatrix 的每个试验进行补零
    for i = 1:length(restMatrix)
        trial_length = size(restMatrix{i}, 1);
        if trial_length < max_rest_length
            restMatrix{i} = [restMatrix{i}; zeros(max_rest_length - trial_length, size(restMatrix{i}, 2))];
        end
    end

    % 对 miMatrix 的每个试验进行补零
    for i = 1:length(miMatrix)
        trial_length = size(miMatrix{i}, 1);
        if trial_length < max_mi_length
            miMatrix{i} = [miMatrix{i}; zeros(max_mi_length - trial_length, size(miMatrix{i}, 2))];
        end
    end

    % 预分配输出变量
    [~, num_rest_trials] = size(restMatrix);
    [~, num_mi_trials] = size(miMatrix);
    pe_rest = cell(1, num_rest_trials);
    pe_rest_spectrum = cell(1, num_rest_trials);
    pe_rest_famp = cell(1, num_rest_trials);

    pe_mi = cell(1, num_mi_trials);
    pe_mi_spectrum = cell(1, num_mi_trials);
    pe_mi_famp = cell(1, num_mi_trials);

    % 处理每个 rest 试验
    for i = 1:num_rest_trials
        [pe_rest{i}, pe_rest_spectrum{i}, pe_rest_famp{i}] = preprocess_trial(restMatrix{i});
    end

    % 处理每个 mi 试验
    for i = 1:num_mi_trials
        [pe_mi{i}, pe_mi_spectrum{i}, pe_mi_famp{i}] = preprocess_trial(miMatrix{i});
    end
end



function [pe_data, pe_spectrum, pe_freq_amplitude] = preprocess_trial(curr_trial)
    Fs = 256; %                         [Hz] Sampling Frequency
    cutoffHigh = 8; %                   [Hz] High pass component
    cutoffLow = 12; %                   [Hz] Low pass component

    % % Make and use band pass filter
    [B,A] = butter(5,[cutoffHigh/(Fs/2),cutoffLow/(Fs/2)]);
    dataTempFilt = filtfilt(B,A,curr_trial);
    % dataTempFilt = bandpass(curr_trial, [cutoffHigh cutoffLow], Fs);

    % Split the EOG and EEG Data
    EOG = dataTempFilt(:,end-1:end);
    dataTempFilt = dataTempFilt(:,1:end-2);

    % TODO: Add EOG Artifact Removal

    % Spatial Filter
    dataSpaceTempFilt = car(dataTempFilt);
    
    pe_data = dataSpaceTempFilt;

    % Frequency Transform
    [pe_spectrum, pe_freq_amplitude] = fft_with_shift(dataTempFilt, Fs, 1);
    % pe_spectrum = pe_spectrum';
end

% Spatial Filtering for EEG
function [filtered_eeg] = car(eeg)
    average_signal = mean(eeg, 2);
    filtered_eeg = eeg - average_signal;  % Subtract average from each element
end

function [freqs, fft_shifted] = fft_with_shift(signal, sample_rate, axis)
    [num_samples, N] = size(signal);
    fft_shifted = fftshift(fft(signal), axis);
    dt = 1/sample_rate; 
    df = 1/dt/(length(signal)-1); 
    freqs = -1/dt/2:df:1/dt/2; 
end

function [freqs, reconstructed_signal] = ifft_with_shift(fft_data, sr)
    reconstructed_signal = ifft(ifftshift(fft_data)); 
    freqs = 0;
end

function [all_sessions] = create_classes(gdfFiles) 
    [num_files,temp]=size(gdfFiles);
    all_sessions =  {};

    for i=1:num_files
        file_chosen = gdfFiles{i};
        file_split = strsplit(file_chosen,"\");
        % disp(file_split{end});
        session_split = strsplit(file_split{end},"_");
        curr_session = session;
        curr_session.Subject = cell2mat(session_split(2));
        curr_session.Session = cell2mat(session_split(6));
        curr_session.Repetition = cell2mat(session_split(7));
        curr_session.Year = session_split(9);
        temp_month = session_split(10);
        temp_day = session_split(11);
        curr_session.Date = temp_month{1} + "-" + temp_day{1};
        curr_session.Online = cell2mat(session_split(4));
        curr_session.Type = cell2mat(session_split(5));
        curr_session.Filename = file_chosen;
        all_sessions{i} = curr_session;
    end
end