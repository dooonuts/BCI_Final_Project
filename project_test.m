clear;
project_data_folder =  "./bci_project_data/";
allFiles = dir(fullfile(project_data_folder, '**', '*.gdf'));
gdfFiles = fullfile({allFiles.folder}, {allFiles.name})';
% Ensure output is a 1D cell array (transpose if necessary)
gdfFiles = gdfFiles(:);

curr_session = session;
curr_session.Date = "11-24-2024"
curr_session.Year = 2023

stuff = 1;


%% TODO LIST
% Finish Session Variable
% Finish PCA
% Signal Cropping
% Create Model and Test
% Linear Discriminant Analysis/Linear Regression
% What happens if we train with first only session of top of that?

%%
file_chosen = gdfFiles{1};
file_split = strsplit(file_chosen,"/");
disp(file_split{end});

[s,h] = sload(file_chosen);

Fs = 256; %                         [Hz] Sampling Frequency
cutoffHigh = 8; %                   [Hz] High pass component
cutoffLow = 12; %                   [Hz] Low pass component

%Certain channels are unused:
s = s(:,1:34);

% %Make and use band pass filter
[B,A] = butter(5,[cutoffHigh/Fs,cutoffLow/Fs]);
dataTempFilt = filtfilt(B,A,s);

% dataTempFilt = bandpass(s, [cutoffHigh cutoffLow], Fs);

%Split the EOG and EEG Data
EOG = dataTempFilt(:,end-1:end);
dataTempFilt = dataTempFilt(:,1:end-2);

% figure(1);clf;
% plot(s(:,1),'b-'); hold on;
% plot(dataTempFilt(:,1),'k-');
%% EOG Artifact Removal - Regression

%Source: Schlogl et. al. Clinical Neuropsychology 2007
%Calculate EOG weights using the cross-covariance matrices
% 
% b = inv(EOG'*EOG)*(EOG'*dataTempFilt);
% 
% dataTempEOG = dataTempFilt - EOG*b;

%% Spatial Filtering - Canonical Correlation Analysis (CCA)

dataSpaceTempFilt = car(dataTempFilt);
% 
% figure(1);clf;
% plot(dataTempFilt(:,2),'b-'); hold on;
% plot(dataSpaceTempFilt(:,2),'k-');

%% Frequency Transform
[num_samples, num_channels] = size(dataSpaceTempFilt);
channel_plotted = 1;

% % FFT Spatial and Time Filtered
% [freqs, freqSpaceTempFilt] = fft_with_shift(dataSpaceTempFilt, Fs, 1);
% figure(3); clf;
% plot(freqs, abs(freqSpaceTempFilt(:,channel_plotted)),'b-');
% 
% % Raw data fft
% figure(5); clf;
% [freqs, freqS] = fft_with_shift(s, Fs, 1);
% plot(freqs, abs(freqS(:,channel_plotted)),'b-');
% 
% % FFT Temporal Filter
% figure(6); clf;
% [freqs, freqTempFilt] = fft_with_shift(dataTempFilt, Fs, 1);
% plot(freqs(:), abs(freqTempFilt(:,channel_plotted)),'b-');

%% PCA 
coeff = pca(dataSpaceTempFilt);


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
