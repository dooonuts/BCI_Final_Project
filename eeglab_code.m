[ALLEEG, EEG, CURRENTSET, ALLCOM] = eeglab;

folder = "./EEGLAB_INPUT/EEG/"
allFiles = dir(fullfile(folder, '**', '*.mat'));
mat_files = fullfile({allFiles.folder}, {allFiles.name})';
mat_files = mat_files(:);


channel_file = './EEGLAB_INPUT/eeg_location_data.ced';
save_path = 'path_to_save/';

for i = 1:12
    temp = strsplit(mat_files{i},"/");
    temp_name = "./"+strjoin(temp(end-2:end),"/");
    srate = 256;
    temp_data = load(mat_files{i});

    % Load Data and Channels
    EEG = pop_importdata('setname', temp{end}, 'data', temp_data.session_data', ...
        'srate', srate, 'nbchan', 34,'chanlocs','./EEGLAB_INPUT/eeg_location_data.ced');
    % Run ICA
    EEG = pop_runica(EEG, 'extended', 1);  % Use all components
    
    EEG = pop_iclabel(EEG, 'default');  % Use default settings
    
    disp('ICLabel Classifications:');
    disp({EEG.etc.ic_classification.ICLabel.classes});  % Show class names
    disp(EEG.etc.ic_classification.ICLabel.classifications);  % Show classification scores
    
    artifact_threshold = 0.5;  % Set threshold for artifact classification
    classifications = EEG.etc.ic_classification.ICLabel.classifications;
    
    %%
    artifact_components = find(classifications(:, 2) > artifact_threshold | ...
        classifications(:, 3) > artifact_threshold | ...
        classifications(:, 4) > artifact_threshold | ...
        classifications(:, 5) > artifact_threshold | ...
        classifications(:, 6) > artifact_threshold | ...
        classifications(:, 7) > artifact_threshold);
    disp(artifact_components);
    
    % EEG = pop_subcomp(EEG, artifact_components, 0);  % Remove identified artifact components
    
    [ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0); % Store dataset in EEGLAB
    pop_saveset(EEG, 'filename', strcat(temp{end},'.set'), 'filepath', './EEGLAB_INPUT/EEG_ICA_REMOVED');
end

%%
temp = strsplit(mat_files{1},"/");
temp_name = "./"+strjoin(temp(end-2:end),"/");
srate = 256;
temp_data = load(mat_files{1});
% Load Data and Channels
EEG = pop_importdata('setname', temp{end}, 'data', temp_data.session_data', ...
    'srate', srate, 'nbchan', 34,'chanlocs','./EEGLAB_INPUT/eeg_location_data.ced');
% Run ICA
EEG = pop_runica(EEG, 'extended', 1);  % Use all components

EEG = pop_iclabel(EEG, 'default');  % Use default settings

disp('ICLabel Classifications:');
disp({EEG.etc.ic_classification.ICLabel.classes});  % Show class names
disp(EEG.etc.ic_classification.ICLabel.classifications);  % Show classification scores

artifact_threshold = 0.5;  % Set threshold for artifact classification
classifications = EEG.etc.ic_classification.ICLabel.classifications;

%%
artifact_components = find(classifications(:, 2) > artifact_threshold | ...
    classifications(:, 3) > artifact_threshold | ...
    classifications(:, 4) > artifact_threshold | ...
    classifications(:, 5) > artifact_threshold | ...
    classifications(:, 6) > artifact_threshold | ...
    classifications(:, 7) > artifact_threshold);
disp(artifact_components);

% EEG = pop_subcomp(EEG, artifact_components, 0);  % Remove identified artifact components

[ALLEEG, EEG, CURRENTSET] = eeg_store(ALLEEG, EEG, 0); % Store dataset in EEGLAB
pop_saveset(EEG, 'filename', strcat(temp{end},'components_removed.set'), 'filepath', './EEGLAB_INPUT/EEG_ICA_REMOVED');


