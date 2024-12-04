clear;
project_data_folder =  "./bci_project_data/";
allFiles = dir(fullfile(project_data_folder, '**', '*.gdf'));
gdfFiles = fullfile({allFiles.folder}, {allFiles.name})';
gdfFiles = gdfFiles(:);

curr_subject = 109; % 107 needs 9000, 108 and 109 can use 8000

num_elements = 10000;
all_sessions = create_classes(gdfFiles);

% Filtering Sessions
EOG_sessions = {};
[~, num_sessions] = size(all_sessions);
for i=1:num_sessions
    curr_session = all_sessions{i};
    if (convertCharsToStrings(all_sessions{i}.Type) == "EOG" & str2num(all_sessions{i}.Subject) == curr_subject)
        [s,h] = sload(curr_session.Filename);
        
        % Certain channels are unused:
        s = s(:,1:34);
        s = butter_filt(s);
        curr_session.Raw_Data = s;
        EOG_sessions{end+1} = curr_session;
    end
end

session_1_data = EOG_sessions{1}.Raw_Data;
session_2_data = EOG_sessions{2}.Raw_Data;

save("./EEGLAB_INPUT/EOG/EOG_DATA_"+num2str(curr_subject)+"_"+EOG_sessions{1}.Session,"session_1_data");
save("./EEGLAB_INPUT/EOG/EOG_DATA_"+num2str(curr_subject)+"_"+EOG_sessions{2}.Session,"session_2_data");

%% 

% Preprocess Each Session and Return, 
function [pe_rest, pe_rest_spectrum, pe_rest_famp, rest_tags, pe_mi, pe_mi_spectrum, pe_mi_famp, mi_tags] = preprocess_session(curr_session_file, num_elements)

    [s,h] = sload(curr_session_file);
    
    % Certain channels are unused:
    s = s(:,1:34);

    [restMatrix,rest_tags,miMatrix,mi_tags] = crop_sort_signals(s,h, num_elements);

    disp(restMatrix);
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

    % No Split for ICA Split the EOG and EEG Data
    % EOG = dataTempFilt(:,end-1:end);
    % dataTempFilt = dataTempFilt(:,1:end-2);

    % TODO: Add EOG Artifact Removal
    
    % ICA needs to happen before spatial filtering, removing spatial filter
    % from this
    
    pe_data = dataTempFilt;

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

function [dataTempFilt] = butter_filt(data)
    Fs = 256; %                         [Hz] Sampling Frequency
    cutoffHigh = 8; %                   [Hz] High pass component
    cutoffLow = 12; %                   [Hz] Low pass component

    % % Make and use band pass filter
    [B,A] = butter(5,[cutoffHigh/(Fs/2),cutoffLow/(Fs/2)]);
    dataTempFilt = filtfilt(B,A,data);

end