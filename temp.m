load("temp.mat");
Fs = 256; %                         [Hz] Sampling Frequency
cutoffHigh = 8; %                   [Hz] High pass component
cutoffLow = 12; %                   [Hz] Low pass component

% % Make and use band pass filter
[B,A] = butter(5,[cutoffHigh/(Fs/2),cutoffLow/(Fs/2)]);
dataTempFilt = filtfilt(B,A,curr_trial);
% dataTempFilt = bandpass(curr_trial, [cutoffHigh cutoffLow], Fs, Steepness=0.95);

% Split the EOG and EEG Data
EOG = dataTempFilt(:,end-1:end);
dataTempFilt = dataTempFilt(:,1:end-2);

% TODO: Add EOG Artifact Removal

pe_data = dataTempFilt;

% Frequency Transform
[pe_spectrum, pe_freq_amplitude] = fft_with_shift(dataTempFilt, Fs, 1);

figure(1); clf;
plot(pe_data)
figure(2); clf;
plot(pe_spectrum,pe_freq_amplitude);


function [freqs, fft_shifted] = fft_with_shift(signal, sample_rate, axis)
    [num_samples, N] = size(signal);
    fft_shifted = fftshift(fft(signal), axis);
    dt = 1/sample_rate; 
    df = 1/dt/(length(signal)-1); 
    freqs = -1/dt/2:df:1/dt/2; 
end