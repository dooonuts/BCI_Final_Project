function [restMatrix,restTags,miMatrix,miTags,channels] = crop_sort_signals(target_s,target_h, num_elements)

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

    %769 is the trigger number for resting action
    rest_event_num = sum(target_h.EVENT.TYP == 769,'all');
    
    %770 is the trigger number for left hand actions
    mi_event_num = sum(target_h.EVENT.TYP == 770, 'all');
    
    %Initialize a cell array - cell mainly because of potentially uneven sizes
    %of each trial
    restMatrix = cell(1,rest_event_num);
    miMatrix = cell(1,mi_event_num);
    
    %Initialize counter for each event type
    rest = 1;
    mi = 1;
    
    %Extract each trial according to left/right hand events
    for n = 1:length(target_h.EVENT.TYP)
    
        %If new event
        if(target_h.EVENT.TYP(n) == 1000)
    
            %Check the tag 2 indexes ahead, if it's 769 we have a right hand
            if(target_h.EVENT.TYP(n+2) == 769)
                tempMatrix = target_s(target_h.EVENT.POS(n):target_h.EVENT.POS(n+4),:);

                restMatrix{rest} = padarray(tempMatrix, [num_elements - size(tempMatrix, 1),0], 0, 'post');

                % restTags{rest} = target_h.EVENT.TYP(n+4);
                temp =  target_h.EVENT.TYP(n+4);

                restTags{rest} = floor(temp/10); % Retrieves type regardless of pass/fail for online
                rest = rest+1;
            %Or if it's 770 we have a left hand
            elseif(target_h.EVENT.TYP(n+2) == 770)
                tempMatrix = target_s(target_h.EVENT.POS(n):target_h.EVENT.POS(n+4),:);
                miMatrix{mi} = padarray(tempMatrix, [num_elements - size(tempMatrix, 1),0], 0, 'post');

                % miTags{mi} = target_h.EVENT.TYP(n+4);
                temp = target_h.EVENT.TYP(n+4);
                miTags{mi} = floor(temp/10);
                mi = mi+1; 
            end
    
        end
    
    end

    %Extracting header information
    channels = h.Label;

end