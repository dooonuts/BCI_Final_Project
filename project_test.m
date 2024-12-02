clear;
project_data_folder =  "./bci_project_data/";
allFiles = dir(fullfile(project_data_folder, '**', '*.gdf'));
gdfFiles = fullfile({allFiles.folder}, {allFiles.name})';
% Ensure output is a 1D cell array (transpose if necessary)
gdfFiles = gdfFiles(:);

repetitions = ['r001';'r002';'r003';'r004'];
sessions = ['s001';'s002';'s003'];
subjects = [107;108;109];

num_elements = 10000;

all_sessions = create_classes(gdfFiles);

% Filtering Sessions
MI_sessions = {};
[~, num_sessions] = size(all_sessions);
for i=1:num_sessions
    if (convertCharsToStrings(all_sessions{i}.Type) == "MI" & str2num(all_sessions{i}.Subject) == 109)
        curr_session = all_sessions{i};
        [pe_rest, pe_rest_spectrum, pe_rest_famp, rest_tags, pe_mi, pe_mi_spectrum, pe_mi_famp, mi_tags] = preprocess_session(curr_session.Filename, num_elements);
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

%% Reshaping Data for PCA and Classification

offline_mi_sessions = {};
online_mi_sessions = {};

num_elements = 10000;
num_trials = 10;
num_channels = 32;

[~, num_mi_sessions] = size(MI_sessions);
for i=1:num_mi_sessions
    curr_session = MI_sessions{i};
    if(convertCharsToStrings(MI_sessions{i}.Online) == "Online")
        temp_session  = reshape_sessions(curr_session, num_elements, num_channels);
        online_mi_sessions{end+1} = temp_session; % 10,(num_elements*num_channels)
    else
        temp_session  = reshape_sessions(curr_session, num_elements, num_channels);
        offline_mi_sessions{end+1} = temp_session;
    end 
end

% Concatenate all sessions
[~, num_online_sessions] = size(online_mi_sessions);
[~, num_offline_sessions] = size(offline_mi_sessions);

total_online_mi_famp= [];
total_online_mi_tags = []; % becomes 120x1 without transpose with vertcat
total_online_rest_famp = [];
total_online_rest_tags = [];

total_offline_mi_famp= [];
total_offline_mi_tags = []; % becomes 120x1 without transpose with vertcat
total_offline_rest_famp = [];
total_offline_rest_tags = [];
 
% First cat offline 
for i=1:num_offline_sessions
    total_offline_mi_famp =  vertcat(total_offline_mi_famp,offline_mi_sessions{i}.PE_MI_Famp);
    total_offline_mi_tags = vertcat(total_offline_mi_tags, cell2mat(offline_mi_sessions{i}.MI_Tags)'); 

    total_offline_rest_famp = vertcat(total_offline_rest_famp, offline_mi_sessions{i}.PE_Rest_Famp);
    total_offline_rest_tags = vertcat(total_offline_rest_tags, cell2mat(offline_mi_sessions{i}.Rest_Tags)');
end

total_offline_sessions = vertcat(total_offline_mi_famp, total_offline_rest_famp); % Gives 80x320000;
total_offline_tags = vertcat(total_offline_mi_tags, total_offline_rest_tags); % Gives 80x1;

% Then cat online
for i=1:num_online_sessions
    total_online_mi_famp = vertcat(total_online_mi_famp,online_mi_sessions{i}.PE_MI_Famp);
    total_online_mi_tags = vertcat(total_online_mi_tags, cell2mat(online_mi_sessions{i}.MI_Tags)');

    total_online_rest_famp = vertcat(total_online_rest_famp, online_mi_sessions{i}.PE_Rest_Famp);
    total_online_rest_tags = vertcat(total_online_rest_tags, cell2mat(online_mi_sessions{i}.Rest_Tags)');
end

total_online_sessions = vertcat(total_online_mi_famp, total_online_rest_famp); % Gives 160x320000;
total_online_tags = vertcat(total_online_mi_tags, total_online_rest_tags); % Gives 160x1;

total_sessions = vertcat(total_offline_sessions, total_online_sessions); % Should give 240x320000;
total_tags = vertcat(total_offline_tags, total_online_tags); % Should give 240x1;

total_sessions = total_sessions'; % Need to do this for PCA

%% PCA
% Should still be split  by trial so can separate later and then run
% training, need to add real to make it not complex
compressed_total_sessions = pca(real(total_sessions)); % Gives 240x240 - trials should still be second axis
compressed_total_sessions = compressed_total_sessions'; % Gives 240x240 - trials should be in first axis - matching total tags

%% Splitting Dataset And Splitting PCA Features
% Get Num PCA Features
num_features = 240; % Total is 240
compressed_total_sessions_num_features = compressed_total_sessions(:,1:num_features);

% Cross Fold/Split Data
% 2x is because rest vs mi; Split Offline vs Online
compressed_offline_sessions = compressed_total_sessions_num_features(1:2*num_offline_sessions*num_trials,:);
compressed_online_sessions = compressed_total_sessions_num_features((2*num_offline_sessions*num_trials)+1:end,:);

%% Training
% Train an LDA classifier on offline data, test on online 
lda_model = fitcdiscr(compressed_offline_sessions, total_offline_tags);

% Predict on test data
y_pred = predict(lda_model, compressed_online_sessions);

% Evaluate accuracy
accuracy = sum(y_pred == total_online_tags) / length(total_online_tags);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

%% TODO LIST
% Create Model and Test
% Linear Discriminant Analysis/Linear Regression
% What happens if we train with first only session of top of that?

%% EOG Artifact Removal - Regression

%Source: Schlogl et. al. Clinical Neuropsychology 2007
%Calculate EOG weights using the cross-covariance matrices
% 
% b = inv(EOG'*EOG)*(EOG'*dataTempFilt);
% 
% dataTempEOG = dataTempFilt - EOG*b;

%% Model 

% Preprocess Each Session and Return, 
function [pe_rest, pe_rest_spectrum, pe_rest_famp, rest_tags, pe_mi, pe_mi_spectrum, pe_mi_famp, mi_tags] = preprocess_session(curr_session_file, num_elements)

    [s,h] = sload(curr_session_file);
    
    % Certain channels are unused:
    s = s(:,1:34);

    [restMatrix,rest_tags,miMatrix,mi_tags] = crop_sort_signals(s,h, num_elements);

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

function [updated_session] = reshape_sessions(curr_session, num_elements, num_channels)
    [~,num_trials] = size(curr_session.PE_MI_Famp);
    updated_session = curr_session;
    curr_PE_MI_Famp = zeros(num_trials,num_elements*32);
    curr_PE_Rest_Famp = zeros(num_trials,num_elements*32);
    
    for i=1:num_trials
        temp_mi =  curr_session.PE_MI_Famp{i};
        temp_rest =  curr_session.PE_Rest_Famp{i};

        padded_mi_floor = padarray(temp_mi, [floor((num_elements - size(temp_mi, 1))/2), 0], 0, 'pre');
        padded_mi = padarray(padded_mi_floor, [ceil((num_elements - size(temp_mi, 1))/2), 0], 0, 'post');

        padded_rest_floor = padarray(temp_rest, [floor((num_elements - size(temp_rest, 1))/2), 0], 0, 'pre');
        padded_rest = padarray(padded_rest_floor, [ceil((num_elements - size(temp_rest, 1))/2), 0], 0, 'post');

        % padded_mi = padarray(temp_mi, [num_elements - size(temp_mi, 1), num_channels - size(temp_mi, 2)], 0, 'post');
        % padded_rest = padarray(temp_rest, [num_elements - size(temp_rest, 1), num_channels - size(temp_rest, 2)], 0, 'post');
        X_2D_mi = reshape(padded_mi, num_elements*num_channels, [])';  % Transpose to make it N x M
        X_2D_rest = reshape(padded_rest, num_elements*32, [])';  % Transpose to make it N x M
        curr_PE_MI_Famp(i,:) = X_2D_mi;
        curr_PE_Rest_Famp(i,:) = X_2D_rest;
    end

    updated_session.PE_MI_Famp = curr_PE_MI_Famp;
    updated_session.PE_Rest_Famp = curr_PE_Rest_Famp;
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
        file_split = strsplit(file_chosen,"/");
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