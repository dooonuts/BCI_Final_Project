% load("temp.mat");
% Fs = 256; %                         [Hz] Sampling Frequency
% cutoffHigh = 8; %                   [Hz] High pass component
% cutoffLow = 12; %                   [Hz] Low pass component
% 
% % % Make and use band pass filter
% [B,A] = butter(5,[cutoffHigh/(Fs/2),cutoffLow/(Fs/2)]);
% dataTempFilt = filtfilt(B,A,curr_trial);
% % dataTempFilt = bandpass(curr_trial, [cutoffHigh cutoffLow], Fs, Steepness=0.95);
% 
% % Split the EOG and EEG Data
% EOG = dataTempFilt(:,end-1:end);
% dataTempFilt = dataTempFilt(:,1:end-2);
% 
% % TODO: Add EOG Artifact Removal
% 
% pe_data = dataTempFilt;
% 
% % Frequency Transform
% [pe_spectrum, pe_freq_amplitude] = fft_with_shift(dataTempFilt, Fs, 1);
% 
% figure(1); clf;
% plot(pe_data)
% figure(2); clf;
% plot(pe_spectrum,pe_freq_amplitude);
% 
% 
% function [freqs, fft_shifted] = fft_with_shift(signal, sample_rate, axis)
%     [num_samples, N] = size(signal);
%     fft_shifted = fftshift(fft(signal), axis);
%     dt = 1/sample_rate; 
%     df = 1/dt/(length(signal)-1); 
%     freqs = -1/dt/2:df:1/dt/2; 
% end

% Load or generate example EEG data (channels x time points x trials)
% Example data: 32 channels, 1000 time points, 50 trials
data = randn(32, 1000, 50);  % (32 x 1000 x 50)
labels = randi([0, 1], 1, 50);  % Labels for each trial (1 x 50)

% Reshape data from 3D to 2D (channels * time points x trials)
[channels, time_points, trials] = size(data); % channels = 32, time_points = 1000, trials = 50
reshaped_data = reshape(data, channels * time_points, trials);  % (32*1000 x 50) => (32000 x 50)

% Perform PCA across all trials (transposing the data to trials x features for PCA)
[coeff, score, latent] = pca(reshaped_data');  % Input size: (50 x 32000), Output score size: (50 x 50)

% - coeff: Principal component coefficients (32000 x 50)
% - score: PCA-transformed data (50 x 50) where each row is a trial
% - latent: Variance explained by each principal component (50 x 1)

% Use the first 'n' principal components (e.g., min(trials, 10) components)
num_components = min(trials, 10);  % Use 10 components or fewer if trials are fewer
pca_transformed_data = score(:, 1:num_components)';  % Select top components and transpose: (10 x 50)

% Check sizes
disp(['Original Data Size: ', mat2str(size(data))]);                      % (32 x 1000 x 50)
disp(['Reshaped Data Size: ', mat2str(size(reshaped_data))]);             % (32000 x 50)
disp(['PCA Transformed Data Size: ', mat2str(size(pca_transformed_data))]); % (10 x 50)
disp(['Labels Size: ', mat2str(size(labels))]);                           % (1 x 50)

% Split PCA-transformed data into train and test sets (e.g., 80% train, 20% test)
num_train = round(0.8 * trials);  % Number of training trials
train_data = pca_transformed_data(:, 1:num_train);  % Training data (10 x 40 if 80%)
test_data = pca_transformed_data(:, num_train+1:end);  % Testing data (10 x 10 if 20%)
train_labels = labels(1:num_train);  % Training labels (1 x 40)
test_labels = labels(num_train+1:end);  % Testing labels (1 x 10)

% Train a simple classifier (e.g., SVM)
Mdl = fitcsvm(train_data', train_labels);  % Input size: train_data' (40 x 10), train_labels (40 x 1)

% Test classifier on test data
predicted_labels = predict(Mdl, test_data');  % Input size: test_data' (10 x 10)

% Evaluate performance
accuracy = sum(predicted_labels == test_labels') / length(test_labels);  % Calculate accuracy
disp(['Classification Accuracy: ', num2str(accuracy)]);


%%
% Define the variable
modelType = 'lda'; % Example variable, can be 'lda', 'svm', or 'mlp'

% Compare the variable using if-elseif statements
if strcmp(modelType, 'lda')
    disp('You selected Linear Discriminant Analysis (LDA).');
elseif strcmp(modelType, 'svm')
    disp('You selected Support Vector Machine (SVM).');
elseif strcmp(modelType, 'mlp')
    disp('You selected Multi-Layer Perceptron (MLP).');
else
    disp('Unknown model type selected.');
end
