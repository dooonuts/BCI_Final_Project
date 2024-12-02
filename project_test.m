clear;
project_data_folder =  "./bci_project_data/";
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

%% Testing Classifier

offline_mi_sessions = {};
online_mi_sessions = {};

[~, num_mi_sessions] = size(MI_sessions);
for i=1:num_mi_sessions
    if(convertCharsToStrings(MI_sessions{i}.Online) == "Online")
        curr_session = MI_sessions{i};
        [~,num_trials] = size(curr_session.PE_MI_Famp);
        curr_PE_MI_Famp = zeros(10,num_elements*32);
        curr_PE_Rest_Famp = zeros(10,num_elements*32);
        
        for i=1:num_trials
            temp_mi =  pca(abs(curr_session.PE_MI_Famp{i}));
            temp_rest =  pca(abs(curr_session.PE_Rest_Famp{i}));
            X_2D_mi = reshape(temp_mi, num_elements*32, [])';  % Transpose to make it N x M
            X_2D_rest = reshape(temp_rest, num_elements*32, [])';  % Transpose to make it N x M
            curr_PE_MI_Famp(i,:) = X_2D_mi;
            curr_PE_Rest_Famp(i,:) = X_2D_rest;
        end

        % curr_session.PE_MI_Famp = curr_PE_MI_Famp;
        % curr_session.PE_Rest_Famp = curr_PE_Rest_Famp;
        online_mi_sessions{end+1} = curr_session; % 10x32x32
    else
        offline_mi_sessions{end+1} = MI_sessions{i};
    end 

end

%%

total_trials = vertcat(curr_PE_MI_Famp, curr_PE_Rest_Famp);
total_tags = horzcat(cell2mat(online_mi_sessions{1}.MI_Tags), cell2mat(online_mi_sessions{1}.Rest_Tags));

% train on offline, test on online
% stuff = fitclinear(online_mi_sessions{1}.PE_MI_Famp, cell2mat(online_mi_sessions{1}.MI_Tags));
% stuff = fitclinear(zeros(10,32), cell2mat(online_mi_sessions{1}.MI_Tags));

% Load or define EEG data
% X: (num_samples x num_features) matrix of features (EEG data)
% y: (num_samples x 1) vector of labels (1 or 2)

% Example initialization (replace this with your actual data)
% X = randn(100, 64); % 100 trials, 64 features per trial
% y = randi([1, 2], 100, 1); % Random labels (1 or 2)

% % Split the data into training and testing sets
% cv = cvpartition(size(X, 1), 'HoldOut', 0.3);
% X_train = X(training(cv), :);
% y_train = y(training(cv));
% X_test = X(test(cv), :);
% y_test = y(test(cv));
% 
% Train an LDA classifier
lda_model = fitcdiscr(total_trials, total_tags);

% Predict on test data
y_pred = predict(lda_model, total_trials)';

% Evaluate accuracy
accuracy = sum(y_pred == total_tags) / length(total_tags);
fprintf('Accuracy: %.2f%%\n', accuracy * 100);

%% Testing 
% first_trial = curr_session.PE_MI{1};
% first_trial_spectrum = curr_session.PE_MI_Spectrum;
% first_trial_famp = curr_session.PE_MI_Famp;
% figure(1); clf;
% plot(first_trial);
% figure(2); clf;
% plot(first_trial_spectrum{1},first_trial_famp{1});



%% TODO LIST
% Finish PCA
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