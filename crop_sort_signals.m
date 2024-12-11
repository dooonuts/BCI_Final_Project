function [restMatrix,restTags,miMatrix,miTags,restFreq,restAmp,miFreq,miAmp] = crop_sort_signals(target_s,target_h, window_size,Fs)

    %Usage: After doing sload, use this function with the resulting data
    %(target_s) and headers (target_h)
    % num_elements = pad_size

    %Output: 
    %   restMatrix: cell matrix of however many rest trials were found in
    %   the data (should be 1x10)
    %   restTags: 1x10 matrix of the result of every rest trial (7692 for
    %   fail, 7693 for success)
    %   miMatrix: cell matrix of however many MI trials were found in
    %   the data (should also be 1x10)
    %   miTags: 1x10 matrix of the result of every MI trial (7702 for fail,
    %   7703 for success)
    channel_labels = target_h.Label(1:end-3);
    save("labels.mat","channel_labels");

    %769 is the trigger number for resting action
    rest_event_num = sum(target_h.EVENT.TYP == 769,'all');
    
    %770 is the trigger number for left hand actions
    mi_event_num = sum(target_h.EVENT.TYP == 770, 'all');
    
    %Initialize a cell array - cell mainly because of potentially uneven sizes
    %of each trial
    restMatrix = {};
    miMatrix = {};
    
    %Initialize counter for each event type
    restTrial = 1;
    miTrial = 1;
    restTags = [];
    miTags = [];
    
    %Extract each trial according to left/right hand events
    for n = 1:length(target_h.EVENT.TYP)
    
        %If new event
        if(target_h.EVENT.TYP(n) == 1000)
    
            %Check the tag 2 indexes ahead, if it's 769 we have a right hand
            if(target_h.EVENT.TYP(n+2) == 769)
                tempMatrix = target_s(target_h.EVENT.POS(n):target_h.EVENT.POS(n+4),:);
                %restMatrix{rest} = padarray(tempMatrix, [num_elements - size(tempMatrix, 1),0], 0, 'post');     
                
                %Find how many windows can fit in a given stretch of data
                numWindows = floor(2*size(tempMatrix,1)/window_size);

                for m = 1:numWindows

                    if(m < numWindows)
                        tempWindow = tempMatrix((m-1)*window_size/2 + 1:(m-1)*window_size/2 + window_size,:);
                        restMatrix{restTrial,m} = car(tempWindow);
                    else
                        tempWindow = tempMatrix((m-1)*window_size/2 + 1:end,:);
                        restMatrix{restTrial,m} = car(padarray(tempWindow,[window_size - size(tempWindow,1),0], 0, 'post'));
                    end

                    % restTags{rest} = target_h.EVENT.TYP(n+4);
                    temp =  target_h.EVENT.TYP(n+4);
    
                    restTags{m} = floor(temp/10); % Retrieves type regardless of pass/fail for online
                    [restFreq{restTrial,m}, restAmp{restTrial,m}] = fft_with_shift(restMatrix{restTrial,m},Fs,1);

                end
            restTrial = restTrial + 1;    
            %Or if it's 770 we have a left hand
            elseif(target_h.EVENT.TYP(n+2) == 770)
                tempMatrix = target_s(target_h.EVENT.POS(n):target_h.EVENT.POS(n+4),:);
                %miMatrix{mi} = padarray(tempMatrix, [num_elements - size(tempMatrix, 1),0], 0, 'post');

                %Find how many windows can fit in a given stretch of data
                numWindows = floor(2*size(tempMatrix,1)/window_size);

                for m = 1:numWindows

                    if(m < numWindows)
                        tempWindow = tempMatrix((m-1)*window_size/2 + 1:(m-1)*window_size/2 + window_size,:);
                        miMatrix{miTrial,m} = car(tempWindow);
                    else
                        tempWindow = tempMatrix((m-1)*window_size/2 + 1:end,:);
                        miMatrix{miTrial,m} = car(padarray(tempWindow,[window_size - size(tempWindow,1),0], 0, 'post'));
                    end

                    [miFreq{miTrial,m}, miAmp{miTrial,m}] = fft_with_shift(miMatrix{miTrial,m},Fs,1);

                    % miTags{mi} = target_h.EVENT.TYP(n+4);
                    temp = target_h.EVENT.TYP(n+4);
                    miTags{m} = floor(temp/10);

                end

                miTrial = miTrial + 1;


            end
    
        end
    
    end

end

function [freqs, fft_shifted] = fft_with_shift(signal, sample_rate, axis)
    [num_samples, N] = size(signal);
    fft_shifted = fftshift(fft(signal), axis);
    dt = 1/sample_rate; 
    df = 1/dt/(length(signal)-1); 
    freqs = -1/dt/2:df:1/dt/2; 
end

% Spatial Filtering for EEG
function [filtered_eeg] = car(eeg)
    average_signal = mean(eeg, 2);
    filtered_eeg = eeg - average_signal;  % Subtract average from each element
end