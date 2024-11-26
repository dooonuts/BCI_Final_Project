clear;
[s,h] = sload('Subject_108_BikeMI_Online_MI_s001_r001_2024_10_18_153644.gdf');

Fs = 256; %                         [Hz] Sampling Frequency
cutoffHigh = 8; %                   [Hz] High pass component
cutoffLow = 12; %                   [Hz] Low pass component

%Certain channels are unused:
s = s(:,1:34);

%Make and use band pass filter
[B,A] = butter(5,[cutoffHigh/Fs,cutoffLow/Fs]);
dataTempFilt = filtfilt(B,A,s);

%% EOG Artifact Removal - Regression

%Source: Schlogl et. al. Clinical Neuropsychology 2007

%Calculate EOG weights using the cross-covariance matrices
EOG = dataTempFilt(:,end-1:end);
dataTempFilt = dataTempFilt(:,1:end-2);

b = inv(EOG'*EOG)*(EOG'*dataTempFilt);

dataTempEOG = dataTempFilt - EOG*b;

%% Spatial Filtering - Canonical Correlation Analysis (CCA)

