clear;
% project_data_folder =  "./bci_project_data/";
project_data_folder = "C:\lls_university\Linshen_pc\UT_Austin\fall2024\ECE385JBIOENG\final_project\BCI Project Data\";
allFiles = dir(fullfile(project_data_folder, '**', '*.gdf'));
gdfFiles = fullfile({allFiles.folder}, {allFiles.name})';
% Ensure output is a 1D cell array (transpose if necessary)
gdfFiles = gdfFiles(:);

repetitions = ['r001';'r002';'r003';'r004'];
sessions = ['s001';'s002';'s003'];
subjects = [107;108;109];

% HyperParameters
curr_subject = 108; 
num_elements = 10000;
num_trials = 10;
num_channels = 32;
num_frequencies = 10000; % Performs frequency cutout after bandpass for easier entry, should only need like 1000 frequencies for this
SHUFFLE_FLAG = true;
PCA_FLAG = true;
num_features = 235; % Total is 240 for 1 subject
k=8; % Number of Folds

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

total_sessions = vertcat(total_offline_sessions, total_online_sessions); % Should give 240x320000;
total_tags = vertcat(total_offline_tags, total_online_tags); % Should give 240x1;

total_sessions = total_sessions'; % Need to do this for PCA

%% PCA
if(PCA_FLAG)
    % Should still be split  by trial so can separate later and then run
    % training, need to add real to make it not complex
    [compressed_total_sessions,scoreTrain,~,~,explained,mu] = pca(real(total_sessions)); % Gives 240x240 - trials should still be second axis
    compressed_total_sessions = compressed_total_sessions'; % Gives 240x240 - trials should be in first axis - matching total tags
    % compressed_total_sessions = compressed_total_sessions; % Gives 240x240 - trials should be in second axis - matching total tags
    
    % Splitting Dataset And Splitting PCA Features
    % Get Num PCA Features
    data = compressed_total_sessions(:,1:num_features);
else
    data = total_sessions'; % Can never run it without PCA b/c too many features
end

%% Cross Fold/Split Data + Shuffle
[num_total_trials,~] = size(data);
cv = cvpartition(num_total_trials,'KFold',k);
labels=total_tags; % Note labels should be 240x1 by here, data should be 240xN;

if(SHUFFLE_FLAG)
    [data, labels] = shuffle_arrays(data, labels);
end

% labels(randperm(length(labels))); % Random permutation of labels for seeing what "chance" is
mean_accuracy = 0;
% Perform k-fold cross-validation
for fold = 1:k
    trainIdx = training(cv, fold);  % Training set indices
    testIdx = test(cv, fold);  % Test set indices

    training_data = data(trainIdx, :);  % Training data
    prediction_data = data(testIdx, :);    % Test data
    training_labels = labels(trainIdx);  % Training labels
    prediction_labels = labels(testIdx);    % Test labels

<<<<<<< HEAD
% Evaluate accuracy
accuracy = sum(y_pred == total_online_tags) / length(total_online_tags);
fprintf('Accuracy: %.2f%%\n', accuracy * 100); % 63.12% here is the accuracy for Daniel's work
%% PCA Linshen's block 
% 1 PCA demo 
data = total_sessions'; % 
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
%% after PCA 58 feature for classification 
labels = zeros(240, 1); 
labels(1:40) = 0;
labels(81:160) = 0;
labels(41:80) = 1;
labels(161:240) = 1;

reduced_data_sequence_class0 = reduced_data([1:40, 81:160], :); %
reduced_data_sequence_class1 = reduced_data([41:80, 161:240], :); % 


%% splite the real and Image part 
% abstract the final version 
first_200_samples = reduced_data_sequence_class0; % 120 x 58 complex matrix
last_200_samples = reduced_data_sequence_class1; % 120 x 58 complex matrix

real_part_first = real(first_200_samples); 
imag_part_first = imag(first_200_samples); 

% abstract last_200_samples real part and image part 
real_part_last = real(last_200_samples); % abstract real 
imag_part_last = imag(last_200_samples); % abstract image 

% check result 
disp('Real part of first_200_samples:');
disp(real_part_first);

disp('Imaginary part of first_200_samples:');
disp(imag_part_first);

disp('Real part of last_200_samples:');
disp(real_part_last);

disp('Imaginary part of last_200_samples:');
disp(imag_part_last);

% real and image part, together.
feature_first_new = [real_part_first, imag_part_first]; % size 120 x 116
feature_last_new = [real_part_last, imag_part_last];   % size 120 x 116


%% visulization, 116 class 
% variable 1 and variable 2 belongs t0 two different class 0 and 1.
first_200_samples = feature_first_new; % first 200 sample 
last_200_samples = feature_last_new; % last 200 sample 

% combine the 200 sample as a matrix 
% combined_samples = [first_200_samples, last_200_samples]; % 这里还需要改成两个不同的
combined_samples = [];
for i = 1:116
    combined_samples = [combined_samples; first_200_samples(:, i)'; last_200_samples(:, i)'];
end
combined_samples=combined_samples'

% boxplot
figure; % can no visulize the complexity value here.
boxplot(combined_samples, 'Labels', arrayfun(@(x) ['Variable ', num2str(x)], 1:232, 'UniformOutput', false));

% legend and title 
legend({'First 200 Samples', 'Last 200 Samples'}, 'Location', 'best'); % the legned does not shows clearly  
%% 3D visulization 

% there are no such different between the first and second layer
% combine data with t-SNE
all_samples = [first_200_samples; last_200_samples];  % combine the data

% tSNE 
tsne_result = tsne(all_samples, 'NumDimensions', 3);

% the data after tSNE 
first_200_tsne = tsne_result(1:120, :);  % first 200 sample 3D dataset 
last_200_tsne = tsne_result(121:240, :); % last 200 sample 3D dataset 

% draw the figure.
figure;
scatter3(first_200_tsne(:, 1), first_200_tsne(:, 2), first_200_tsne(:, 3), 50, 'r', 'filled');
hold on;
scatter3(last_200_tsne(:, 1), last_200_tsne(:, 2), last_200_tsne(:, 3), 50, 'b', 'filled');
xlabel('t-SNE 1');
ylabel('t-SNE 2');
zlabel('t-SNE 3');
title('3D t-SNE Visualization of First 200 vs Last 200 Samples');
legend({'First 200 Samples', 'Last 200 Samples'}, 'Location', 'best');
grid on;
%% MLP classification 
 k = 15;
[MLP_c1_matrix_106] = MLPTrainingWithKFold(a1vsa2vsa3, MAV_Labels_a1vsa2vsa3, k);

%% MLP result 
plotConfusionMatrixAndAccuracy(MLP_c1_matrix_106, 106);
%% 2 SVM 
k = 15;
[SVM_c1_matrix_106] = SVMTrainingWithKFold_SVM(a1vsa2vsa3, MAV_Labels_a1vsa2vsa3, k);

%% 2 SVM result 

plotConfusionMatrixAndAccuracy(SVM_c1_matrix_106, 106);

%% 3 Linear method  

k = 15;  
[li_c1_matrix_106] = LinearTrainingWithKFold(a1vsa2vsa3, MAV_Labels_a1vsa2vsa3, k);
%% 3 Linear model 
plotConfusionMatrixAndAccuracy(li_c1_matrix_106, 106);

%% 4 XGBoost 
[all_error_items_XGBoost, XGBoost_c1_matrix_106] = XGBoostClassification(a1vsa2vsa3, MAV_Labels_a1vsa2vsa3, k); 

%% 4 XGBoost display 
plotConfusionMatrixAndAccuracy(XGBoost_c1_matrix_106, 106);

%% 5 Random forest 
[all_error_items_Random_forest, RF_c1_matrix_106] = randomForestClassification(a1vsa2vsa3, MAV_Labels_a1vsa2vsa3, k);

%% 5 Random Forest display 
plotConfusionMatrixAndAccuracy(RF_c1_matrix_106, 106);



=======
    lda_model = fitcdiscr(training_data, training_labels);
    linear_pred = predict(lda_model, prediction_data);
    [accuracy,~,~] = plotConfusionMatrix(prediction_labels, linear_pred, true);

    mean_accuracy = mean_accuracy+accuracy;
end
mean_accuracy = mean_accuracy/k;
disp("Mean Accuracy: " + num2str(mean_accuracy));

%% Online vs Offline 2x is because rest vs mi;
% compressed_offline_sessions = compressed_total_sessions_num_features(1:2*num_offline_sessions*num_trials,:);
% compressed_online_sessions = compressed_total_sessions_num_features((2*num_offline_sessions*num_trials)+1:end,:);
% 
% training_data = compressed_offline_sessions;
% training_labels = total_offline_tags;
% prediction_data = compressed_online_sessions;
% prediction_labels = total_online_tags;


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
>>>>>>> 64970886109ea8e8b5cadfefebf66d30f3f2cfa5

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

<<<<<<< HEAD
% post preprocessing draw 
function plotConfusionMatrixAndAccuracy(confMatrix, subjectNum)
    % confuse matrix 
    figure;
    confusionchart(confMatrix);
    title(['Confusion Matrix for Subject ' num2str(subjectNum)]);
    
    % get TP、FN、FP and TN
    TP = confMatrix(1, 1); % (True Positive)
    FN = confMatrix(1, 2); % (False Negative)
    FP = confMatrix(2, 1); % (False Positive)
    TN = confMatrix(2, 2); % (True Negative)
    
    % accuracy 
    accuracy = (TP + TN) / (TP + TN + FP + FN);
    
    % show accuracy 
    disp(['Subject ' num2str(subjectNum) ' Accuracy:']);
    disp(accuracy);
end

% 1 MLP 

function [linear_c1_matrix_106] = MLPTrainingWithKFold(a1vsa2vsa3, MAV_Labels_a1vsa2vsa3, k)
    % MLPTrainingWithKFold: 使用K-fold交叉验证训练MLP模型并计算混淆矩阵
    %
    % 输入:
    %   a1vsa2vsa3 - 特征数据矩阵 (num_samples x num_features)
    %   MAV_Labels_a1vsa2vsa3 - 标签向量 (num_samples x 1)
    %   k - K-fold交叉验证的折数
    %
    % 输出:
    %   linear_c1_matrix_106 - 累积的混淆矩阵 (2x2)

    % 初始化混淆矩阵
    linear_c1_matrix_106 = zeros(2, 2);

    % 交叉验证分割数据
    linear_c1 = cvpartition(length(MAV_Labels_a1vsa2vsa3), 'KFold', k);

    % 遍历每一折进行训练和测试
    for i = 1:k
        disp(sprintf("Iteration no: %d", i)); 
        
        % 使用 fitcnet 训练 MLP 模型
        MLP_Model_106 = fitcnet(a1vsa2vsa3(linear_c1.training(i),:), ...
                                MAV_Labels_a1vsa2vsa3(linear_c1.training(i)), ...
                                'LayerSizes', [200 200 200], ...       % 网络层大小
                                'Activations', 'relu', ...            % 激活函数
                                'Standardize', true, ...              % 数据标准化
                                'Lambda', 2e-5);                     % 正则化参数

        % 测试集预测
        test = a1vsa2vsa3(linear_c1.test(i), :);
        predictedLabels1 = predict(MLP_Model_106, test);
        
        % 计算混淆矩阵
        linear_confused1 = confusionmat(MAV_Labels_a1vsa2vsa3(linear_c1.test(i)), predictedLabels1);
        
        % 如果混淆矩阵是2x2的，则累积结果
        [numRows1, ~] = size(linear_confused1);
        if numRows1 == 2
            linear_c1_matrix_106 = linear_c1_matrix_106 + linear_confused1;
        end
    end
end


% 2 SVM 

function [linear_c1_matrix_106] = SVMTrainingWithKFold_SVM(a1vsa2vsa3, MAV_Labels_a1vsa2vsa3, k)
    % SVMTrainingWithKFold_SVM: 使用K-fold交叉验证训练SVM模型并计算混淆矩阵
    %
    % 输入:
    %   a1vsa2vsa3 - 特征数据矩阵 (num_samples x num_features)
    %   MAV_Labels_a1vsa2vsa3 - 标签向量 (num_samples x 1)
    %   k - K-fold交叉验证的折数
    %
    % 输出:
    %   linear_c1_matrix_106 - 累积的混淆矩阵 (2x2)

    % 初始化混淆矩阵
    linear_c1_matrix_106 = zeros(2, 2);

    % 交叉验证分割数据
    linear_c1 = cvpartition(length(MAV_Labels_a1vsa2vsa3), 'KFold', k);

    % 遍历每一折进行训练和测试
    for i = 1:k-1
        disp(sprintf("Iteration no: %d", i)); 
        
        % 使用 fitcsvm 训练 SVM 模型
        SVM_Model_106 = fitcsvm(a1vsa2vsa3(linear_c1.training(i),:), ...
                                MAV_Labels_a1vsa2vsa3(linear_c1.training(i)), ...
                                'KernelFunction', 'rbf', ...          % 径向基核函数
                                'BoxConstraint', 1, ...              % 正则化参数
                                'Standardize', true, ...             % 数据标准化
                                'KernelScale', 'auto');              % 自动调整核函数参数
        
        % 测试集预测
        test = a1vsa2vsa3(linear_c1.test(i), :);
        predictedLabels1 = predict(SVM_Model_106, test);
        
        % 计算混淆矩阵
        linear_confused1 = confusionmat(MAV_Labels_a1vsa2vsa3(linear_c1.test(i)), predictedLabels1);
        
        % 如果混淆矩阵是2x2的，则累积结果
        [numRows1, ~] = size(linear_confused1);
        if numRows1 == 2
            linear_c1_matrix_106 = linear_c1_matrix_106 + linear_confused1;
        end
    end
end



% 3 LiNear Model 
function [linear_c1_matrix_106] = LinearTrainingWithKFold(a1vsa2vsa3, MAV_Labels_a1vsa2vsa3, k)
    linear_c1_matrix_106 = zeros(2, 2);
    linear_c1 = cvpartition(length(MAV_Labels_a1vsa2vsa3), 'KFold', k);
    for i = 1:k-1
        disp(sprintf("Iteration no: %d", i)); 
        
 
        Linear_Model_106 = fitclinear(a1vsa2vsa3, MAV_Labels_a1vsa2vsa3, ...
                                      'Learner', 'logistic', ...          
                                      'Regularization', 'ridge', ...     
                                      'Lambda', 1e-4);                    
        
        test = a1vsa2vsa3(linear_c1.test(i), :);
        predictedLabels1 = predict(Linear_Model_106, test);      
        linear_confused1 = confusionmat(MAV_Labels_a1vsa2vsa3(linear_c1.test(i)), predictedLabels1);       
        [numRows1, ~] = size(linear_confused1);
        if numRows1 == 2
            linear_c1_matrix_106 = linear_c1_matrix_106 + linear_confused1;
        end
    end
end



% 4 XGBoost 

function [all_error_items, confusionMatrix] = XGBoostClassification(a1vsa2vsa3, MAV_Labels_a1vsa2vsa3, k)

    confusionMatrix = zeros(2, 2);
    all_error_items = [];
    

    linear_c1 = cvpartition(length(MAV_Labels_a1vsa2vsa3), 'KFold', k);

    for i = 1:k
        disp(sprintf("Iteration no: %d", i)); 
        

        XGB_Model_106 = fitcensemble(a1vsa2vsa3(linear_c1.training(i), :), ...
                                     MAV_Labels_a1vsa2vsa3(linear_c1.training(i)), ...
                                     'Method', 'LogitBoost', ...  
                                     'NumLearningCycles', 100, ... 
                                     'Learners', templateTree('MaxNumSplits', 20)); 
        

        test = a1vsa2vsa3(linear_c1.test(i), :);
        trueLabels = MAV_Labels_a1vsa2vsa3(linear_c1.test(i));
        predictedLabels1 = predict(XGB_Model_106, test);
        

        testIndices = find(linear_c1.test(i)); 
        error_indices = testIndices(predictedLabels1' ~= trueLabels); 
        all_error_items = [all_error_items; error_indices]; 


        linear_confused1 = confusionmat(MAV_Labels_a1vsa2vsa3(linear_c1.test(i)), predictedLabels1);
        
        [numRows1, ~] = size(linear_confused1);
        if numRows1 == 2
            confusionMatrix = confusionMatrix + linear_confused1;
        end
    end
end




% 5 Random forest 

function [all_error_items, confusionMatrix] = randomForestClassification(a1vsa2vsa3, MAV_Labels_a1vsa2vsa3, k)
    
    % inital con
    confusionMatrix = zeros(2, 2);
    all_error_items = [];
    
    % Cross-validation split data
    linear_c1 = cvpartition(length(MAV_Labels_a1vsa2vsa3), 'KFold', k);

    for i = 1:k
        disp(sprintf("Iteration no: %d", i));
        
        % Random Forest training 
        RF_Model_106 = TreeBagger(50, ... % number of tree 
                                  a1vsa2vsa3(linear_c1.training(i), :), ...
                                  MAV_Labels_a1vsa2vsa3(linear_c1.training(i)), ...
                                  'Method', 'classification', ...
                                  'NumPredictorsToSample', 'all', ...
                                  'OOBPrediction', 'On');
        
        % test strategy training 
        test = a1vsa2vsa3(linear_c1.test(i), :);
        trueLabels = MAV_Labels_a1vsa2vsa3(linear_c1.test(i));
        predictedLabels1 = str2double(predict(RF_Model_106, test)); % output as str
        
        % output item number 
        testIndices = find(linear_c1.test(i)); % test item number 
        error_indices = testIndices(predictedLabels1' ~= trueLabels); % Wrong tag index
        all_error_items = [all_error_items; error_indices]; % Aggregate Error Number
        
        % Calculate the confusion matrix 
        linear_confused1 = confusionmat(MAV_Labels_a1vsa2vsa3(linear_c1.test(i)), predictedLabels1);
        
        [numRows1, ~] = size(linear_confused1);
        if numRows1 == 2
            confusionMatrix = confusionMatrix + linear_confused1;
        end
    end
end

 
=======
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

>>>>>>> 64970886109ea8e8b5cadfefebf66d30f3f2cfa5
