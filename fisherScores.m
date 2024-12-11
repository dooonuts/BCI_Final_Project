function [bands,scores] = fisherScores(freq,restAmp,miAmp,numBands)

    binLength = floor(length(freq)/numBands);
    
    elRest = [];
    elMI = [];
    scores = zeros(32,numBands);
    %Extraction and reshaping
    for sess = 1:length(session)

        currSessionRest = restAmp{sess};
        currSessionMI = miAmp{sess};

        windowsRest = length(currSessionRest);
        windowsMI = length(currSessionMI);

        for win = 1:windowsRest
           elRest(:,:,win) = restAmp{sess}{win};
        end

        for win = 1:windowsMI
           elMI(:,:,win) = miAmp{sess}{win};
        end
    end

    %Calculate the power
    elRestPower = abs(elRest);
    elMIPower = abs(elMI);

    for n = 1:numBands
        
        %Define bin Lower and Upper Limits
        binLL = (n-1)*binLength+1;
        binUL = n*binLength;
        bands{n} = [freq(binLL),freq(binUL)];
        %Take mean of class 1
        meanRest = mean(elRestPower(binLL:binUL,1:32,:),[1,3]);

        %Take mean of class 2
        meanMI = mean(elMIPower(binLL:binUL,1:32,:),[1,3]);

        %Take stdev of class 1
        sdRest = std(elRestPower(binLL:binUL,1:32,:),0,[1,3]);

        %Take stdev of class 2
        sdMI = std(elMIPower(binLL:binUL,1:32,:),0,[1,3]);

        %Calculate Fisher
        scores(:,n) = (meanRest - meanMI).^2 ./ (sqrt(sdRest) + sqrt(sdMI));
    end
end

function windows = reshapeFrequencyWindows(session)

    for sess = 1:length(session)
        currSession = session{sess};
        numWin = length(currSession);
        for win = 1:numWin
            windows(:,:,win) = session{sess}{win};
        end
    end

end