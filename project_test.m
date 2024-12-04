clear;
project_data_folder =  "./bci_project_data/";
allFiles = dir(fullfile(project_data_folder, '**', '*.gdf'));
gdfFiles = fullfile({allFiles.folder}, {allFiles.name})';
gdfFiles = gdfFiles(:);

repetitions = ['r001';'r002';'r003';'r004'];
sessions = ['s001';'s002';'s003'];
subjects = [107;108;109];

% HyperParameters
curr_subject = 107; % 107 needs 9000, 108 and 109 can use 8000
num_elements = 9000;
num_trials = 10;
num_channels = 32;
num_frequencies = 8000; % Performs frequency cutout after bandpass for easier entry, should only need like 1000 frequencies for this
SHUFFLE_FLAG = true;
PCA_FLAG = true;
num_features = 90; % Total is 239 for 1 subject
k=20; % Number of Folds
model_type = 'lda'; % Choice between lda, svm, mlp

RANDOM_FLAG = false;

all_sessions = create_classes(gdfFiles);

% Filtering Sessions
MI_sessions = {};
[~, num_sessions] = size(all_sessions);
for i=1:num_sessions
    if (convertCharsToStrings(all_sessions{i}.Type) == "MI" & str2num(all_sessions{i}.Subject) == curr_subject)
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

[~, num_mi_sessions] = size(MI_sessions);
for i=1:num_mi_sessions
    curr_session = MI_sessions{i};
    if(convertCharsToStrings(MI_sessions{i}.Online) == "Online")
        temp_session  = reshape_sessions(curr_session, num_frequencies, num_channels);
        online_mi_sessions{end+1} = temp_session; % 10,(num_frequencies*num_channels)
    else
        temp_session  = reshape_sessions(curr_session, num_frequencies, num_channels);
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
% Testing only offline sessions
% total_sessions = total_offline_sessions; % Should give 240x320000;
% total_tags = total_offline_tags; % Should give 240x1;
% Testing only online sessions
% total_sessions = total_online_sessions; % Should give 240x320000;
% total_tags = total_online_tags; % Should give 240x1;

total_sessions = vertcat(total_offline_sessions, total_online_sessions); % Should give 240x320000;
total_tags = vertcat(total_offline_tags, total_online_tags); % Should give 240x1;

% total_sessions = total_sessions'; % Need to do this for PCA

%% PCA
if(PCA_FLAG)
    % Should still be split  by trial so can separate later and then run
    % training, need to add real to make it not complex
    [compressed_total_sessions,scoreTrain,latent,tsquared,explained,mu] = pca(total_sessions); % Gives 240x240 - trials should first axis
    % [compressed_total_sessions,scoreTrain,~,~,explained,mu] = pca(total_sessions); % Gives 240x240 - trials should still be second axis
    % compressed_total_sessions = compressed_total_sessions'; % Gives 240x240 - trials should be in first axis - matching total tags
    % compressed_total_sessions = compressed_total_sessions; % Gives 240x240 - trials should be in second axis - matching total tags
    
    % Splitting Dataset And Splitting PCA Features
    % Get Num PCA Features
    % Select the number of components that explain 95% of the variance
    explained_variance = cumsum(explained) / sum(explained);
    num_components = find(explained_variance >= 0.95, 1); % Select the first component meeting the threshold
    disp("95% variance from this many components: "+num2str(num_components));
    % data = compressed_total_sessions(1:num_features,:)';
    data = real(scoreTrain(:,1:num_features));
else
    data = total_sessions; % Can never run it without PCA b/c too many features
end

%% Cross Fold/Split Data + Shuffle
[num_total_trials,~] = size(data);
cv = cvpartition(num_total_trials,'KFold',k);
labels=total_tags; % Note labels should be 240x1 by here, data should be 240xN;

if(SHUFFLE_FLAG)
    [data, labels] = shuffle_arrays(data, labels);
end

if(RANDOM_FLAG)
    labels(randperm(length(labels))); % Random permutation of labels for seeing what "chance" is
end

mean_accuracy = 0;
% Perform k-fold cross-validation
for fold = 1:k
    trainIdx = training(cv, fold);  % Training set indices
    testIdx = test(cv, fold);  % Test set indices

    training_data = data(trainIdx, :);  % Training data
    training_labels = labels(trainIdx);  % Training labels
    prediction_data = data(testIdx, :);    % Test data
    prediction_labels = labels(testIdx);    % Test labels
    accuracy = 0;
    if strcmp(model_type,'lda')
        [accuracy,~,~] = lda(training_data,training_labels,prediction_data,prediction_labels);
    elseif strcmp(model_type,'svm')
        [accuracy,~,~] = svm(training_data,training_labels,prediction_data,prediction_labels);
    elseif strcmp(model_type,'mlp')
        [accuracy,~,~] = mlp(training_data,training_labels,prediction_data,prediction_labels);
    else
        disp("Wrong Model Type!")
        break;
    end
    % [accuracy,~,~] = lda(training_data,training_labels,prediction_data,prediction_labels);
    disp(accuracy);
    mean_accuracy = mean_accuracy+accuracy;
end
mean_accuracy = mean_accuracy/k;
disp("Mean Accuracy: " + num2str(mean_accuracy));

%% Online vs Offline 2x is because rest vs mi;
% compressed_offline_sessions = compressed_total_sessions_num_features(1:2*num_offline_sessions*num_trials,:);
% compressed_online_sessions = compressed_total_sessions_num_features((2*num_offline_sessions*num_trials)+1:end,:);


% compressed_offline_sessions = data(1:2*num_offline_sessions*num_trials,:);
% compressed_online_sessions = data(2*num_offline_sessions*num_trials+1:end,:);
% 
% training_data = compressed_offline_sessions;
% training_labels = total_offline_tags;
% prediction_data = compressed_online_sessions;
% prediction_labels = total_online_tags;
% 
% [accuracy,~,~] = lda(training_data,training_labels,prediction_data,prediction_labels);
% disp("Online Accuracy : " + num2str(accuracy));

% Looks like shuffling the data doesn't matter for performance either way,
% if given same online/offline data it works so I think this is correct

% disp("LDA: ")
% % Linear Discriminant Analysis/Linear Regression
% % Train an LDA classifier on offline data, test on online 
% lda_model = fitcdiscr(training_data, training_labels);
% linear_pred = predict(lda_model, prediction_data);
% plotConfusionMatrix(prediction_labels, linear_pred, true);
% 
% disp("SVM: ")
% % SVM
% svm_model=fitcsvm(training_data,training_labels,'KernelFunction', 'linear');
% svm_pred = predict(svm_model, prediction_data);
% plotConfusionMatrix(prediction_labels, svm_pred, true);
% 
% disp("MLP")
% mlp_model = fitcnet(training_data,training_labels,'LayerSizes', [5 10]);
% mlp_pred = predict(mlp_model,prediction_data);
% plotConfusionMatrix(prediction_labels, mlp_pred, true);

% accuracy = sum(y_pred == total_online_tags) / length(total_online_tags);
% fprintf('Accuracy: %.2f%%\n', accuracy * 100);

%% TODO LIST
% EOG Artifact Removal
% K-Fold Cross Validation
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

%% Extra Functions

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
    [pe_spectrum, pe_freq_amplitude] = fft_with_shift(dataSpaceTempFilt, Fs, 1);
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

function [updated_session] = reshape_sessions(curr_session, num_frequencies, num_channels)
    [~,num_trials] = size(curr_session.PE_MI_Famp);
    updated_session = curr_session;
    curr_PE_MI_Famp = zeros(num_trials,num_frequencies*num_channels);
    curr_PE_Rest_Famp = zeros(num_trials,num_frequencies*num_channels);
    
    for i=1:num_trials
        temp_mi =  curr_session.PE_MI_Famp{i};
        temp_rest =  curr_session.PE_Rest_Famp{i};
        temp_mi_spectrum = curr_session.PE_MI_Spectrum{i};
        temp_rest_spectrum = curr_session.PE_Rest_Spectrum{i};

        % Adding frequency cropping
        temp_mi = cropCenter(temp_mi,num_frequencies);
        temp_rest = cropCenter(temp_rest,num_frequencies);
        temp_mi_spectrum = cropCenter(temp_mi_spectrum',num_frequencies);
        temp_rest_spectrum = cropCenter(temp_rest_spectrum',num_frequencies);

        updated_session.PE_MI_Spectrum{i} = temp_mi_spectrum';
        updated_session.PE_Rest_Spectrum{i} = temp_rest_spectrum';
        
        X_2D_mi = reshape(temp_mi, num_frequencies*num_channels, [])';  % Transpose to make it N x M
        X_2D_rest = reshape(temp_rest, num_frequencies*num_channels, [])';  % Transpose to make it N x M
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

function [accuracy,precision,recall] = plotConfusionMatrix(actual, predicted, plot)
    % Create and display the confusion chart with raw counts only
    cm = confusionchart(actual, predicted, ...
        'RowSummary', 'off', ...          % Turn off row summary (no percentages)
        'ColumnSummary', 'off', ...       % Turn off column summary (no percentages)
        'DiagonalColor', [0 0.6 0.2], ... % Green diagonal for correct predictions
        'OffDiagonalColor', [0.8 0.2 0.2]); % Red off-diagonal for misclassifications

    % Additional customization (optional)
    cm.Title = 'Confusion Matrix (Raw Counts)';
    cm.XLabel = 'Predicted Class';
    cm.YLabel = 'Actual Class';
    matrix=cm.NormalizedValues;

    TP = matrix(1, 1);  % True Positives
    FN = matrix(1, 2);  % False Negatives
    FP = matrix(2, 1);  % False Positives
    TN = matrix(2, 2);  % True Negatives
    accuracy = (TP + TN) / (TP + TN + FP + FN);
    % disp("subject accuracy: " + num2str(accuracy));
    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    % disp("precision: " + num2str(precision));
    % disp("recall: " + num2str(recall));
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

function [A_shuffled, B_shuffled] = shuffle_arrays(A, B)
    numRows = size(A, 1);

    shuffled_idx = randperm(numRows);  % Random permutation of row indices

    A_shuffled = A(shuffled_idx, :);
    B_shuffled = B(shuffled_idx, :);
end

% Crops rows not columns
function croppedMatrix = cropCenter(matrix, num_frequencies)
    [numRows, numCols] = size(matrix);

    if numRows <= num_frequencies
        croppedMatrix = matrix;
    else   
        startIdx = floor((numRows - num_frequencies) / 2) + 1;
        
        croppedMatrix = matrix(startIdx:startIdx + num_frequencies - 1, :);
    end
end

function [accuracy, precision, recall,linear_pred] = lda(training_data, training_labels, prediction_data, prediction_labels)
    lda_model = fitcdiscr(training_data, training_labels);
    %lda_model = fitclinear(training_data, training_labels);
    linear_pred = predict(lda_model, prediction_data);
    [accuracy,precision,recall] = plotConfusionMatrix(prediction_labels, linear_pred, true);
end

function [accuracy, precision, recall, svm_pred] = svm(training_data, training_labels, prediction_data, prediction_labels)
    svm_model=fitcsvm(training_data,training_labels,'KernelFunction', 'linear');
    svm_pred = predict(svm_model, prediction_data);
    [accuracy,precision,recall] = plotConfusionMatrix(prediction_labels, svm_pred, true);
end

function [accuracy, precision, recall, mlp_pred] = mlp(training_data, training_labels, prediction_data, prediction_labels)
    mlp_model = fitcnet(training_data,training_labels,'LayerSizes', [5 10]);
    mlp_pred = predict(mlp_model,prediction_data);
    [accuracy,precision,recall] = plotConfusionMatrix(prediction_labels, mlp_pred, true);
end