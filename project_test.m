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
        [curr_data, curr_spectrum, curr_famp] = preprocess_session(curr_session.Filename);
        curr_session.PEData = curr_data;
        curr_session.Spectrum = curr_spectrum;
        curr_session.Famp = curr_famp;
        MI_sessions{end+1} = curr_session;
    end
end

% [curr_data, curr_spectrum, curr_famp] = preprocess_session(MI_sessions{1}.Filename);

%% TODO LIST
% Finish Session Variable
% Finish PCA
% Signal Cropping
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
function [pe_data, pe_spectrum, pe_freq_amplitude] = preprocess_session(curr_session_file)

    [s,h] = sload(curr_session_file);
    
    Fs = 256; %                         [Hz] Sampling Frequency
    cutoffHigh = 8; %                   [Hz] High pass component
    cutoffLow = 12; %                   [Hz] Low pass component
    
    % Certain channels are unused:
    s = s(:,1:34);
    
    % Make and use band pass filter
    [B,A] = butter(5,[cutoffHigh/Fs,cutoffLow/Fs]);
    dataTempFilt = filtfilt(B,A,s);

    % TODO: Add Crop in Time Domain

    % Split the EOG and EEG Data
    EOG = dataTempFilt(:,end-1:end);
    dataTempFilt = dataTempFilt(:,1:end-2);

    % TODO: Add EOG Artifact Removal

    % Spatial Filter
    dataSpaceTempFilt = car(dataTempFilt);
    
    pe_data = dataSpaceTempFilt;

    % Frequency Transform
    [pe_spectrum, pe_freq_amplitude] = fft_with_shift(dataTempFilt, Fs, 1);
    pe_spectrum = pe_spectrum';
    
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
        % disp(file_split{end});
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