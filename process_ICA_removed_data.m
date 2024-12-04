project_data_folder =  "./bci_project_data/";
allFiles = dir(fullfile(project_data_folder, '**', '*.gdf'));
gdfFiles = fullfile({allFiles.folder}, {allFiles.name})';
gdfFiles = gdfFiles(:);

ica_folder =  "./EEGLAB_INPUT/EEG_ICA_REMOVED/";
icaFiles = dir(fullfile(ica_folder, '**', '*.set'));
icaFiles = fullfile({icaFiles.folder}, {icaFiles.name})';
icaFiles = icaFiles(:);

%% Testing Loading
% curr_ica_file = './EEGLAB_INPUT/EEG_ICA_REMOVED/EEG_DATA_107_s001_Offline_r001.mat.set'
% curr_ica_file = find_ica_file(icaFiles, {curr_session.Session, curr_session.Repetition, curr_session.Online});
% EEG = pop_loadset('filename', curr_ica_file);
% data = EEG.data';

%% 
curr_subject = 107; % 107 needs 9000, 108 and 109 can use 8000
num_elements = 10000;
num_trials = 10;
num_channels = 32;
num_frequencies = 10000; % Performs frequency cutout after bandpass for easier entry, should only need like 1000 frequencies for this

all_sessions = create_classes(gdfFiles);
% Filtering Sessions
MI_sessions = {};
[~, num_sessions] = size(all_sessions);
for i=1:num_sessions
    if (convertCharsToStrings(all_sessions{i}.Type) == "MI" & str2num(all_sessions{i}.Subject) == curr_subject)
        curr_session = all_sessions{i};
        [pe_rest, pe_rest_spectrum, pe_rest_famp, rest_tags, pe_mi, pe_mi_spectrum, pe_mi_famp, mi_tags] = preprocess_ica_session(curr_session.Filename, curr_session, icaFiles, num_elements);
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

%% Extra Functions

% Preprocess Each Session and Return, 
function [pe_rest, pe_rest_spectrum, pe_rest_famp, rest_tags, pe_mi, pe_mi_spectrum, pe_mi_famp, mi_tags] = preprocess_ica_session(curr_session_file, curr_session, ica_files, num_elements)

    [s,h] = sload(curr_session_file);

    curr_ica_file = find_ica_file(ica_files, {curr_session.Session, curr_session.Repetition, curr_session.Online});
    EEG = pop_loadset('filename', curr_ica_file);
    
    % Override s with new thing
    s = EEG.data';

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