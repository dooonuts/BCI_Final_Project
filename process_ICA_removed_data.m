project_data_folder =  "C:\Users\thien\Documents\BCI Project Data\";
allFiles = dir(fullfile(project_data_folder, '**', '*.gdf'));
gdfFiles = fullfile({allFiles.folder}, {allFiles.name})';
gdfFiles = gdfFiles(:);

ica_folder =  "C:\Users\thien\Documents\BCI Project Data\8-30Hz";
icaFiles = dir(fullfile(ica_folder, '**', '*.set'));
icaFiles = fullfile({icaFiles.folder}, {icaFiles.name})';
icaFiles = icaFiles(:);

%% Testing Loading
% curr_ica_file = './EEGLAB_INPUT/EEG_ICA_REMOVED/EEG_DATA_107_s001_Offline_r001.mat.set'
% curr_ica_file = find_ica_file(icaFiles, {curr_session.Session, curr_session.Repetition, curr_session.Online});
% EEG = pop_loadset('filename', curr_ica_file);
% data = EEG.data';

%% 
curr_subject = 108; % 107 needs 9000, 108 and 109 can use 8000
%num_elements = 10000;
window_size = 256; 
num_trials = 10;
num_channels = 32;
Fs = 256;
num_bands = 128;

all_sessions = create_classes(gdfFiles);
% Filtering Sessions
MI_sessions = {};
[~, num_sessions] = size(all_sessions);
for i=1:num_sessions
    if (convertCharsToStrings(all_sessions{i}.Type) == "MI" & str2num(all_sessions{i}.Subject) == curr_subject)
        curr_session = all_sessions{i};
        [pe_rest, pe_rest_spectrum, pe_rest_famp, rest_tags, pe_mi, pe_mi_spectrum, pe_mi_famp, mi_tags] = preprocess_ica_session(curr_session.Filename, curr_session, icaFiles, window_size,Fs);
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
        online_mi_sessions{end+1} = curr_session; % 10,(num_frequencies*num_channels)

    else
        offline_mi_sessions{end+1} = curr_session;

        %Take Fisher

        %To make my life easier, make a second variable
        temp_MIFamp = reshape(curr_session.PE_MI_Famp,[1,size(curr_session.PE_MI_Famp,1)*size(curr_session.PE_MI_Famp,2)]);
        temp_MIFamp = temp_MIFamp(~cellfun(@isempty,temp_MIFamp));
        temp_RestFamp = reshape(curr_session.PE_Rest_Famp,[1,size(curr_session.PE_Rest_Famp,1)*size(curr_session.PE_Rest_Famp,2)]);
        temp_RestFamp = temp_RestFamp(~cellfun(@isempty,temp_RestFamp));
        offReshapedMIFamps{i} = temp_MIFamp;
        offReshapedRestFamps{i} = temp_RestFamp;

    end 
end

%Find Fisher Scores across offline sessions
offReshapedMIFamps = offReshapedMIFamps(~cellfun(@isempty,offReshapedMIFamps));
offReshapedRestFamps = offReshapedRestFamps(~cellfun(@isempty,offReshapedRestFamps));
[bands,offFullFisher] = fisherScores(pe_rest_spectrum{1,1},offReshapedRestFamps,offReshapedMIFamps,numBands);

figure(99);
offFullFisher = offFullFisher(:,end/2:end);
imagesc(0:Fs/numBands:Fs/2,1:32,offFullFisher);
xlabel('Frequency (Hz)')
ylabel('Channel');

channames = ["Fp1"; "Fpz"; "Fp2"; "F7"; "F3"; "Fz"; "F4"; "F8"; "FC5"; "FC1"; "FC2"; "FC6";...
"M1"; "T7"; "C3"; "Cz"; "C4"; "T8"; "M2"; "CP5"; "CP1"; "CP2"; "CP6"; "P7"; "P3"; "Pz"; "P4"; "P8";...
"POz"; "O1"; "Oz"; "O2"];
yticks(1:32);
yticklabels(channames);

fontsize(gca,15,'points');
title(sprintf('All Fisher Scores Across Subject %i',curr_subject));

load selectedChannels.mat


figure(100);
topoplot(offFullFisher(:,9),selectedChannels,'maplimits','maxmin','electrodes','labels');
title(sprintf('Best Fisher Score Channel of Subject %i',curr_subject));
CB = colorbar;
ylabel(CB,'Fisher Score');
fontsize(gca,15,'points');

%% Online vs Offline 2x is because rest vs mi;

%By visual inspection of the Fisher scores, determine which frequency bands
%are needed to train

%Choose bands using the Fisher score
%TO DO: PICK BANDS AND CHANNEL COMBINATIONS USING FISHER

usedBands = [69,70];
bandIndices = [];

for n = 1:length(usedBands)

    bandRanges = bands{usedBands(n)};
    tempBandIndices = [find(pe_mi_spectrum{1,1} == bandRanges(1)) find(pe_mi_spectrum{1,1} == bandRanges(2))];
    bandIndices = [bandIndices tempBandIndices(1):tempBandIndices(2)];
end

offlineMI_res = reshapeFrequencyWindows(offReshapedMIFamps);
offlineRest_res = reshapeFrequencyWindows(offReshapedRestFamps);
offlineMI_res = abs(offlineMI_res(bandIndices,1:32,:));
offlineRest_res = abs(offlineRest_res(bandIndices,1:32,:));

offlineMI_res = permute(offlineMI_res,[3,1,2]);
offlineRest_res = permute(offlineRest_res,[3,1,2]);
offlineMI_res = reshape(offlineMI_res, [size(offlineMI_res,1),size(offlineMI_res,2)*size(offlineMI_res,3)]);
offlineRest_res = reshape(offlineRest_res, [size(offlineRest_res,1),size(offlineRest_res,2)*size(offlineRest_res,3)]);

%Make labels
labelArray = [zeros(size(offlineRest_res,1),1); ones(size(offlineMI_res,1),1)];

%% Labeling Stuff
full_training_data = [offlineMI_res;offlineRest_res];
full_training_labels = labelArray;

validation_indices = randperm(length(labelArray),100);
validation_data = full_training_data(validation_indices,:);
validation_labels = labelArray(validation_indices);

training_data = full_training_data;
training_data(validation_indices,:) = [];
training_labels = labelArray;
training_labels(validation_indices) = [];

%

%%

disp("LDA: ")
% Linear Discriminant Analysis/Linear Regression
% Train an LDA classifier on offline data, test on online 
lda_model = fitcdiscr(training_data, training_labels);
linear_pred = predict(lda_model, validation_data);
figure(1);
[accLDAVal, precLDAVal, recLDAVal,] = plotConfusionMatrix(validation_labels, linear_pred, true);
title('LDA Validation Confusion Matrix');

disp("Logistic Regression: ")
%Logistic Regression
logreg_model = fitglm(training_data,training_labels);
logreg_pred = predict(logreg_model,validation_data);
logreg_pred(logreg_pred > 0.5) = 1;
logreg_pred(logreg_pred <= 0.5) = 0;
logreg_pred = double(logreg_pred);
figure(2);
[accLogRegVal, precLogRegVal, recLogRegVal,] = plotConfusionMatrix(validation_labels, logreg_pred, true);
title('Logistic Regression Validation Confusion Matrix');

disp("SVM: ")
% SVM
svm_model=fitcsvm(training_data,training_labels,'KernelFunction','linear');
svm_pred = predict(svm_model, validation_data);
figure(3);
[accSVMVal, precSVMVal, recSVMVal] = plotConfusionMatrix(validation_labels, svm_pred, true);
title('SVM Validation Confusion Matrix');

disp("MLP")
mlp_model = fitcnet(training_data,training_labels,'LayerSizes', [5 10]);
mlp_pred = predict(mlp_model,validation_data);
figure(4);
[accMLPVal, precMLPVal, recMLPVal] = plotConfusionMatrix(validation_labels, mlp_pred, true);
title('MLP Validation Confusion Matrix');

%% With a trained model, test on online data

%All the online sessions are known to be the last 8
sess = 5:12;
true_labels = [];
predict_labels = [];
trialCounter = 1;

for n = 1:length(sess)

    %Select session
    currSession = MI_sessions{n};
    curr_MI_Famp = currSession.PE_MI_Famp;
    curr_Rest_Famp = currSession.PE_Rest_Famp;

    %Within each session, there are 20 trials, 10 rest, 10 bike
    %Iterate through them all
    for trial = 1:10

        %Do rest here
        true_labels(trialCounter) = 0;
        predict_labels(trialCounter,1) = classifyOnline(curr_Rest_Famp(trial,:),lda_model,bandIndices);
        predict_labels(trialCounter,2) = classifyOnline(curr_Rest_Famp(trial,:),svm_model,bandIndices);
        predict_labels(trialCounter,3) = classifyOnline(curr_Rest_Famp(trial,:),mlp_model,bandIndices);
        predict_labels(trialCounter,4) = classifyOnline(curr_Rest_Famp(trial,:),logreg_model,bandIndices);

        trialCounter = trialCounter + 1;
        %Do MI here
        true_labels(trialCounter) = 1;
        predict_labels(trialCounter,1) = classifyOnline(curr_MI_Famp(trial,:),lda_model,bandIndices);
        predict_labels(trialCounter,2) = classifyOnline(curr_MI_Famp(trial,:),svm_model,bandIndices);
        predict_labels(trialCounter,3) = classifyOnline(curr_MI_Famp(trial,:),mlp_model,bandIndices);
        predict_labels(trialCounter,4) = classifyOnline(curr_MI_Famp(trial,:),logreg_model,bandIndices);
        
        trialCounter = trialCounter + 1;
    end

end

%Confusion matrices
figure(5);
disp("LDA: ")
[accLDATest, precLDATest, recLDATest] = plotConfusionMatrix(true_labels, predict_labels(:,1), true);
title('LDA Testing Confusion Matrix');
disp("Logistic Regression: ")
figure(6);
[accLogRegTest, precLogRegTest, recLogRegTest] = plotConfusionMatrix(true_labels, predict_labels(:,4), true);
title('Logistic Regression Testing Confusion Matrix');
disp("SVM: ")
figure(7);
[accSVMTest, precSVMTest, recSVMTest] = plotConfusionMatrix(true_labels, predict_labels(:,2), true);
title('SVM Testing Confusion Matrix');
figure(8);
disp("MLP: ")
[accMLPTest, precMLPTest, recMLPTest] = plotConfusionMatrix(true_labels, predict_labels(:,3), true);
title('LDA Testing Confusion Matrix');



%% Concatenate all sessions
% [~, num_online_sessions] = size(online_mi_sessions);
% [~, num_offline_sessions] = size(offline_mi_sessions);
% 
% total_online_mi_famp= [];
% total_online_mi_tags = []; % becomes 120x1 without transpose with vertcat
% total_online_rest_famp = [];
% total_online_rest_tags = [];
% 
% total_offline_mi_famp= [];
% total_offline_mi_tags = []; % becomes 120x1 without transpose with vertcat
% total_offline_rest_famp = [];
% total_offline_rest_tags = [];
% 
% % First cat offline 
% for i=1:num_offline_sessions
%     total_offline_mi_famp =  vertcat(total_offline_mi_famp,offline_mi_sessions{i}.PE_MI_Famp);
%     total_offline_mi_tags = vertcat(total_offline_mi_tags, cell2mat(offline_mi_sessions{i}.MI_Tags)'); 
% 
%     total_offline_rest_famp = vertcat(total_offline_rest_famp, offline_mi_sessions{i}.PE_Rest_Famp);
%     total_offline_rest_tags = vertcat(total_offline_rest_tags, cell2mat(offline_mi_sessions{i}.Rest_Tags)');
% end
% 
% total_offline_sessions = vertcat(total_offline_mi_famp, total_offline_rest_famp); % Gives 80x320000;
% total_offline_tags = vertcat(total_offline_mi_tags, total_offline_rest_tags); % Gives 80x1;
% 
% % Then cat online
% for i=1:num_online_sessions
%     total_online_mi_famp = vertcat(total_online_mi_famp,online_mi_sessions{i}.PE_MI_Famp);
%     total_online_mi_tags = vertcat(total_online_mi_tags, cell2mat(online_mi_sessions{i}.MI_Tags)');
% 
%     total_online_rest_famp = vertcat(total_online_rest_famp, online_mi_sessions{i}.PE_Rest_Famp);
%     total_online_rest_tags = vertcat(total_online_rest_tags, cell2mat(online_mi_sessions{i}.Rest_Tags)');
% end
% 
% total_online_sessions = vertcat(total_online_mi_famp, total_online_rest_famp); % Gives 160x320000;
% total_online_tags = vertcat(total_online_mi_tags, total_online_rest_tags); % Gives 160x1;
% % Testing only offline sessions
% % total_sessions = total_offline_sessions; % Should give 240x320000;
% % total_tags = total_offline_tags; % Should give 240x1;
% % Testing only online sessions
% % total_sessions = total_online_sessions; % Should give 240x320000;
% % total_tags = total_online_tags; % Should give 240x1;
% 
% total_sessions = vertcat(total_offline_sessions, total_online_sessions); % Should give 240x320000;
% total_tags = vertcat(total_offline_tags, total_online_tags); % Should give 240x1;
% 
% % total_sessions = total_sessions'; % Need to do this for PCA

%% Extra Functions

% Preprocess Each Session and Return, 
function [pe_rest, pe_rest_spectrum, pe_rest_famp, rest_tags, pe_mi, pe_mi_spectrum, pe_mi_famp, mi_tags] = preprocess_ica_session(curr_session_file, curr_session, ica_files, window_size,Fs)

    [s,h] = sload(curr_session_file);

    curr_ica_file = find_ica_file(ica_files, {curr_session.Session, curr_session.Repetition, curr_session.Online, curr_session.Subject});
    EEG = pop_loadset('filename', curr_ica_file);
    
    % Override s with new thing
    s = EEG.data';

    [pe_rest,rest_tags,pe_mi,mi_tags,pe_rest_spectrum,pe_rest_famp,pe_mi_spectrum,pe_mi_famp] = crop_sort_signals(s,h, window_size,Fs);

    % [num_rest_trials, num_rest_windows] = size(restMatrix);
    % [num_mi_trials, num_mi_windows] = size(miMatrix);
    % pe_rest = {};
    % pe_rest_spectrum = {};
    % pe_rest_famp = {};
    % 
    % pe_mi = {};
    % pe_mi_spectrum = {};
    % pe_mi_famp = {};
    % for i=1:num_rest_trials
    %     for j = 1:num_rest_windows
    %     [pe_rest{i,j},pe_rest_spectrum{i,j},pe_rest_famp{i,j}] = preprocess_trial(restMatrix{i,j});
    %     end
    % end
    % for i=1:num_mi_trials
    %     for j = 1:num_mi_windows
    %         [pe_mi{i,j},pe_mi_spectrum{i,j},pe_mi_famp{i,j}] = preprocess_trial(miMatrix{i,j});
    %     end
    % end 
end

function [pe_data, pe_spectrum, pe_freq_amplitude] = preprocess_trial(curr_trial)
    Fs = 256;
    curr_trial = curr_trial(:,1:end-2);

    % Spatial Filter
    dataSpaceTempFilt = car(curr_trial);
    
    pe_data = dataSpaceTempFilt;

    % Frequency Transform
    [pe_spectrum, pe_freq_amplitude] = fft_with_shift(dataSpaceTempFilt, Fs, 1);
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

function [ica_filename] = find_ica_file(ica_files, substrings)
    disp(substrings)
    [num_ica_files, ~] = size(ica_files);
    for i = 1:num_ica_files
        text = ica_files{i};
        is_present = cellfun(@(sub) contains(text, sub), substrings);
        if(all(is_present))
            ica_filename = text;
        end
    end
end

function [all_sessions] = create_classes(gdfFiles) 
    [num_files,temp]=size(gdfFiles);
    all_sessions =  {};

    for i=1:num_files
        file_chosen = gdfFiles{i};
        file_split = strsplit(file_chosen,"\");
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
    svm_model=fitcsvm(training_data,training_labels,'KernelFunction','');
    svm_pred = predict(svm_model, prediction_data);
    [accuracy,precision,recall] = plotConfusionMatrix(prediction_labels, svm_pred, true);
end

function [accuracy, precision, recall, mlp_pred] = mlp(training_data, training_labels, prediction_data, prediction_labels)
    mlp_model = fitcnet(training_data,training_labels,'LayerSizes', [5 10]);
    mlp_pred = predict(mlp_model,prediction_data);
    [accuracy,precision,recall] = plotConfusionMatrix(prediction_labels, mlp_pred, true);
end

function windows = reshapeFrequencyWindows(session)

    for sess = 1:length(session)
        currSession = session{sess};
        numWin = length(currSession);
        for win = 1:numWin
            windows(:,:,win) = session{sess}{win};
        end
    end

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
    disp("subject accuracy: " + num2str(accuracy));
    precision = TP/(TP+FP);
    recall = TP/(TP+FN);
    % disp("precision: " + num2str(precision));
    % disp("recall: " + num2str(recall));
end

function prediction = classifyOnline(trial,model,bandIndices)

    %Define thresholds
    miThreshold = 0;
    
    %Running total of MI vs non-MI
    conf = 0;

    %Loop through the length of the given cell array of windows
    for n = 1:length(trial)
        
        if(~isempty(trial{n}))
            tempTrial = trial{n};
            tempTrial = tempTrial(bandIndices,1:32);
            tempTrial = reshape(tempTrial',[1,size(tempTrial,1)*size(tempTrial,2)]);
            pred = predict(model,abs(tempTrial));
        
            if(isa(model,"GeneralizedLinearModel"))
                
                if(pred > 0.5)
                    pred = 1;
                else
                    pred = 0;
                end

            end

            if(pred == 1)
                conf = conf + 1;
            else
                conf = conf - 1;
            end

        end
    end

    %At the end of trial looping, classify the whole thing based on
    %confidence, favoring rest if at exactly a 50-50
    if(conf > 0)
        prediction = 1;
    else
        prediction = 0;
    end
    

end